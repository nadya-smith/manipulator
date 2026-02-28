# robot_backend/mujoco_robot.py
"""
MuJoCo-бэкенд робота Rooky Arm.

Загружает XML-модель из пути, указанного в конфиге
(robot.simulation.xml_path).  Имена суставов, актуаторов
и сенсоров берутся из конфига / соглашения об именовании.

Соглашение о сенсорах в XML:
    sens_pos_{1..N}   — положения суставов (jointpos)
    sens_vel_{1..N}   — скорости (jointvel)
    sens_frc_{1..N}   — усилия актуаторов (actuatorfrc)
    tcp_pos, tcp_quat — поза TCP (framepos / framequat)
"""

import os
import numpy as np
import threading
import time
from typing import List, Optional

from .base_robot import BaseRobot, RobotMode, RobotState
from logger import logger
from config import cfg

# ── Проверка доступности MuJoCo ─────────────────────
MUJOCO_AVAILABLE = False
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    logger.add("[MuJoCo] Библиотека не найдена — pip install mujoco")
except OSError as e:
    logger.add(f"[MuJoCo] Ошибка загрузки DLL: {e}")


class MuJoCoRobot(BaseRobot):
    """Симуляция 6-DOF робота Rooky Arm в MuJoCo."""

    def __init__(self):
        super().__init__()
        self._mode = RobotMode.SIMULATION

        # ── Количество суставов из конфига ──
        self._num_joints = cfg.joint_count()  # обычно 6

        # ── Домашняя позиция (градусы → радианы) ──
        home_deg = cfg.get("robot.home_position_deg",
                           [0.0] * self._num_joints)
        self._home_position = [np.radians(d) for d in home_deg]

        # ── Имена камер из конфига ──
        self._cam_overview = cfg.get("mujoco.cameras.overview",
                                     "overview_cam")
        self._cam_work = cfg.get("mujoco.cameras.work", "work_cam")

        # ── Размеры рендера из конфига ──
        self._render_w = cfg.get("mujoco.rendering.width", 640)
        self._render_h = cfg.get("mujoco.rendering.height", 480)

        # ── Параметры схвата из конфига ──
        self._grip_open_rad = cfg.get("gripper.open_rad", 0.0)
        self._grip_close_rad = cfg.get("gripper.close_rad", 1.300)

        # ── MuJoCo-объекты ──
        self._model = None
        self._data = None

        # ── Два рендерера ──
        self._renderer_overview = None
        self._renderer_workcam = None

        # ── Поток симуляции ──
        self._sim_thread = None
        self._sim_running = False
        self._lock = threading.Lock()

        # ── Кэш состояния ──
        self._cache_joint_pos = [0.0] * self._num_joints
        self._cache_joint_vel = [0.0] * self._num_joints
        self._cache_joint_frc = [0.0] * self._num_joints
        self._cache_tcp_pos = [0.0, 0.0, 0.0]
        self._cache_tcp_quat = [1.0, 0.0, 0.0, 0.0]

        # ── Целевые значения ──
        self._target_joints = list(self._home_position)
        self._target_gripper = 0.0
        self._gripper_state = 0.0
        self._paused = False
        self._emergency = False

        # ── ID-шники (заполняются в _resolve_ids) ──
        self._joint_actuator_ids: List[int] = []
        self._gripper_actuator_id: int = -1
        self._joint_pos_sensor_ids: List[int] = []
        self._joint_vel_sensor_ids: List[int] = []
        self._joint_frc_sensor_ids: List[int] = []
        self._tcp_pos_sensor_id: int = -1
        self._tcp_quat_sensor_id: int = -1

        # ── Имена суставов MuJoCo (для IK) ──
        self._mujoco_joint_names: List[str] = []

        # ── Очереди рендер-запросов ──
        self._overview_requested = False
        self._overview_frame = None
        self._overview_event = threading.Event()

        self._workcam_requested = False
        self._workcam_frame = None
        self._workcam_event = threading.Event()

        # ── Флаг создания рендереров ──
        self._create_renderers = True

    # ═══════════════════════════════════════════════════
    #  Свойства
    # ═══════════════════════════════════════════════════

    @property
    def is_connected(self) -> bool:
        return self._model is not None and self._data is not None

    @property
    def is_moving(self) -> bool:
        return any(abs(v) > 0.01 for v in self._cache_joint_vel)

    # ═══════════════════════════════════════════════════
    #  Подключение / Отключение
    # ═══════════════════════════════════════════════════

    def connect(self, target: str = "", **kwargs) -> bool:
        if not MUJOCO_AVAILABLE:
            logger.add("[MuJoCo] Библиотека недоступна!")
            self._state = RobotState.ERROR
            return False

        try:
            # ── Определяем путь к XML ──
            xml_path = target if (target and target.endswith(".xml")) else ""
            if not xml_path:
                xml_path = cfg.get("robot.simulation.xml_path", "")

            # Пробуем разные базовые каталоги
            if xml_path and not os.path.isfile(xml_path):
                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
                alt = os.path.join(project_root, xml_path)
                if os.path.isfile(alt):
                    xml_path = alt

            if not xml_path or not os.path.isfile(xml_path):
                logger.add(
                    f"[MuJoCo] XML-модель не найдена: {xml_path!r}\n"
                    f"         → robot.simulation.xml_path в config")
                self._state = RobotState.ERROR
                return False

            # ── Загрузка модели ──
            self._model = mujoco.MjModel.from_xml_path(xml_path)
            self._data = mujoco.MjData(self._model)
            logger.add(f"[MuJoCo] Модель загружена: {xml_path}")

            # ── Разрешение ID ──
            self._resolve_ids()

            # ── Начальное положение ──
            for i, aid in enumerate(self._joint_actuator_ids):
                if aid >= 0:
                    self._data.ctrl[aid] = self._home_position[i]
            # Прогоняем несколько шагов для стабилизации
            for _ in range(200):
                mujoco.mj_step(self._model, self._data)
            self._update_cache()

            self._create_renderers = kwargs.get("enable_rendering", True)

            # ── Запуск потока симуляции ──
            self._sim_running = True
            self._emergency = False
            self._paused = False
            self._sim_thread = threading.Thread(
                target=self._simulation_loop, daemon=True)
            self._sim_thread.start()

            self._state = RobotState.IDLE
            logger.add("[MuJoCo] Симуляция запущена ✓")
            return True

        except Exception as e:
            logger.add(f"[MuJoCo] Ошибка подключения: {e}")
            import traceback
            logger.add(traceback.format_exc())
            self._state = RobotState.ERROR
            return False

    def disconnect(self) -> None:
        self._sim_running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=3.0)
            self._sim_thread = None
        self._renderer_overview = None
        self._renderer_workcam = None
        self._model = None
        self._data = None
        self._state = RobotState.DISCONNECTED
        logger.add("[MuJoCo] Симуляция остановлена")

    # ═══════════════════════════════════════════════════
    #  Разрешение ID суставов / актуаторов / сенсоров
    # ═══════════════════════════════════════════════════

    def _resolve_ids(self):
        """
        Читает имена суставов и актуаторов из конфига,
        находит их ID в скомпилированной модели.
        Сенсоры ищутся по соглашению: sens_pos_{i}, sens_vel_{i}, sens_frc_{i}.
        """
        m = self._model
        obj = mujoco.mjtObj
        joints_cfg = cfg.get("joints", [])

        # ── Актуаторы суставов ──
        self._joint_actuator_ids = []
        self._mujoco_joint_names = []
        for j in joints_cfg:
            act_name = j.get("mujoco_actuator", "")
            jnt_name = j.get("mujoco_name", "")
            aid = mujoco.mj_name2id(m, obj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                logger.add(f"[MuJoCo] ⚠ Актуатор не найден: {act_name}")
            self._joint_actuator_ids.append(aid)
            self._mujoco_joint_names.append(jnt_name)

        # ── Актуатор схвата ──
        grip_act_name = cfg.get("gripper.mujoco_actuator",
                                "motor_J5_grip")
        self._gripper_actuator_id = mujoco.mj_name2id(
            m, obj.mjOBJ_ACTUATOR, grip_act_name)
        if self._gripper_actuator_id < 0:
            logger.add(f"[MuJoCo] ⚠ Актуатор схвата не найден: "
                        f"{grip_act_name}")

        # ── Сенсоры (по конвенции имён) ──
        n = len(joints_cfg)
        self._joint_pos_sensor_ids = []
        self._joint_vel_sensor_ids = []
        self._joint_frc_sensor_ids = []
        for i in range(1, n + 1):
            self._joint_pos_sensor_ids.append(
                mujoco.mj_name2id(m, obj.mjOBJ_SENSOR, f"sens_pos_{i}"))
            self._joint_vel_sensor_ids.append(
                mujoco.mj_name2id(m, obj.mjOBJ_SENSOR, f"sens_vel_{i}"))
            self._joint_frc_sensor_ids.append(
                mujoco.mj_name2id(m, obj.mjOBJ_SENSOR, f"sens_frc_{i}"))

        self._tcp_pos_sensor_id = mujoco.mj_name2id(
            m, obj.mjOBJ_SENSOR, "tcp_pos")
        self._tcp_quat_sensor_id = mujoco.mj_name2id(
            m, obj.mjOBJ_SENSOR, "tcp_quat")

        # ── Лог ──
        ok_joints = sum(1 for a in self._joint_actuator_ids if a >= 0)
        ok_sens = sum(1 for s in self._joint_pos_sensor_ids if s >= 0)
        logger.add(f"[MuJoCo] Найдено: {ok_joints}/{n} актуаторов, "
                    f"{ok_sens}/{n} сенсоров, "
                    f"TCP={self._tcp_pos_sensor_id >= 0}, "
                    f"Grip={self._gripper_actuator_id >= 0}")

    # ═══════════════════════════════════════════════════
    #  Кэш
    # ═══════════════════════════════════════════════════

    def _update_cache(self):
        sd = self._data.sensordata
        adr = self._model.sensor_adr

        for i, sid in enumerate(self._joint_pos_sensor_ids):
            if sid >= 0:
                self._cache_joint_pos[i] = float(sd[adr[sid]])
        for i, sid in enumerate(self._joint_vel_sensor_ids):
            if sid >= 0:
                self._cache_joint_vel[i] = float(sd[adr[sid]])
        for i, sid in enumerate(self._joint_frc_sensor_ids):
            if sid >= 0:
                self._cache_joint_frc[i] = float(sd[adr[sid]])

        if self._tcp_pos_sensor_id >= 0:
            pa = adr[self._tcp_pos_sensor_id]
            self._cache_tcp_pos = [float(sd[pa + k]) for k in range(3)]

        if self._tcp_quat_sensor_id >= 0:
            qa = adr[self._tcp_quat_sensor_id]
            self._cache_tcp_quat = [float(sd[qa + k]) for k in range(4)]

    # ═══════════════════════════════════════════════════
    #  Главный цикл симуляции
    # ═══════════════════════════════════════════════════

    def _simulation_loop(self):
        dt = self._model.opt.timestep
        logger.add(f"[MuJoCo] Цикл симуляции dt={dt:.4f}с")

        # Рендереры создаём в этом потоке
        if self._create_renderers:
            try:
                self._renderer_overview = mujoco.Renderer(
                    self._model,
                    height=self._render_h,
                    width=self._render_w)
                self._renderer_workcam = mujoco.Renderer(
                    self._model,
                    height=self._render_h,
                    width=self._render_w)
                logger.add("[MuJoCo] Оба рендерера созданы ✓")
            except Exception as e:
                logger.add(f"[MuJoCo] Рендереры недоступны: {e}")
                self._renderer_overview = None
                self._renderer_workcam = None

        step_count = 0

        while self._sim_running:
            if self._paused or self._emergency:
                time.sleep(0.01)
                continue

            with self._lock:
                # ── Управление суставами ──
                for i, aid in enumerate(self._joint_actuator_ids):
                    if aid >= 0:
                        self._data.ctrl[aid] = self._target_joints[i]

                # ── Управление схватом ──
                if self._gripper_actuator_id >= 0:
                    grip_ctrl = (
                        self._grip_open_rad
                        + self._target_gripper
                        * (self._grip_close_rad - self._grip_open_rad)
                    )
                    self._data.ctrl[self._gripper_actuator_id] = grip_ctrl

                # ── Шаг физики ──
                mujoco.mj_step(self._model, self._data)
                step_count += 1

                # Обновляем кэш каждые 5 шагов
                if step_count % 5 == 0:
                    self._update_cache()

                # ── Рендер обзорной камеры ──
                if (self._overview_requested
                        and self._renderer_overview is not None):
                    try:
                        self._renderer_overview.update_scene(
                            self._data, camera=self._cam_overview)
                        rgb = self._renderer_overview.render()
                        self._overview_frame = rgb[:, :, ::-1].copy()
                    except Exception:
                        self._overview_frame = None
                    self._overview_requested = False
                    self._overview_event.set()

                # ── Рендер рабочей камеры ──
                if (self._workcam_requested
                        and self._renderer_workcam is not None):
                    try:
                        self._renderer_workcam.update_scene(
                            self._data, camera=self._cam_work)
                        rgb = self._renderer_workcam.render()
                        self._workcam_frame = rgb[:, :, ::-1].copy()
                    except Exception:
                        self._workcam_frame = None
                    self._workcam_requested = False
                    self._workcam_event.set()

            time.sleep(dt)

    # ═══════════════════════════════════════════════════
    #  Получение кадров (два вида)
    # ═══════════════════════════════════════════════════

    def get_overview_frame(self) -> Optional[np.ndarray]:
        """3D-обзор сцены (таб «Симуляция»)."""
        if not self.is_connected or self._renderer_overview is None:
            return None
        self._overview_event.clear()
        self._overview_requested = True
        if self._overview_event.wait(timeout=0.2):
            return self._overview_frame
        return None

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Рабочая камера (таб «Камера + Детекция»)."""
        if not self.is_connected or self._renderer_workcam is None:
            return None
        self._workcam_event.clear()
        self._workcam_requested = True
        if self._workcam_event.wait(timeout=0.2):
            return self._workcam_frame
        return None

    # ═══════════════════════════════════════════════════
    #  Чтение состояния
    # ═══════════════════════════════════════════════════

    def get_joint_positions(self) -> List[float]:
        return list(self._cache_joint_pos)

    def get_joint_velocities(self) -> List[float]:
        return list(self._cache_joint_vel)

    def get_joint_torques(self) -> List[float]:
        return list(self._cache_joint_frc)

    def get_cartesian_pose(self) -> List[float]:
        pos = self._cache_tcp_pos
        rpy = self._quat_to_euler(np.array(self._cache_tcp_quat))
        return [pos[0], pos[1], pos[2],
                float(rpy[0]), float(rpy[1]), float(rpy[2])]

    # ═══════════════════════════════════════════════════
    #  Движение
    # ═══════════════════════════════════════════════════

    def move_j(self, joint_positions: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        """Движение в конфигурационном пространстве (рад)."""
        if not self.is_connected or self._emergency:
            return False
        self._state = RobotState.MOVING
        self._target_joints = list(
            joint_positions[:self._num_joints])
        if blocking:
            self._wait_until_reached(joint_positions)
        self._state = RobotState.IDLE
        return True

    def move_l(self, cartesian_pose: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        """Линейное движение TCP (простой IK)."""
        if not self.is_connected or self._emergency:
            return False
        self._state = RobotState.MOVING
        target_pos = np.array(cartesian_pose[:3])
        success = self._simple_ik(target_pos)
        self._state = RobotState.IDLE
        return success

    def move_to_home(self) -> bool:
        return self.move_j(self._home_position, blocking=True)

    def stop(self) -> None:
        self._target_joints = list(self._cache_joint_pos)
        self._state = RobotState.IDLE

    def emergency_stop(self) -> None:
        self._emergency = True
        self._target_joints = list(self._cache_joint_pos)
        self._state = RobotState.EMERGENCY
        logger.add("[MuJoCo] ⛔ ЭКСТРЕННАЯ ОСТАНОВКА")

    def pause(self) -> None:
        self._paused = True
        self._state = RobotState.PAUSED

    def resume(self) -> None:
        self._paused = False
        self._emergency = False
        self._state = RobotState.IDLE

    # ═══════════════════════════════════════════════════
    #  Схват
    # ═══════════════════════════════════════════════════

    def set_gripper(self, value: float) -> None:
        """0.0 = открыт, 1.0 = закрыт."""
        self._target_gripper = max(0.0, min(1.0, value))
        self._gripper_state = self._target_gripper

    def get_gripper(self) -> float:
        return self._gripper_state

    # ═══════════════════════════════════════════════════
    #  Информация
    # ═══════════════════════════════════════════════════

    def get_info(self) -> dict:
        return {
            "mode":         self._mode.value,
            "state":        self._state.value,
            "model":        "Rooky Arm 6-DOF (MuJoCo)",
            "connected":    self.is_connected,
            "num_joints":   self._num_joints,
            "sim_time":     float(self._data.time) if self._data else 0.0,
            "has_overview": self._renderer_overview is not None,
            "has_workcam":  self._renderer_workcam is not None,
        }

    # ═══════════════════════════════════════════════════
    #  Утилиты
    # ═══════════════════════════════════════════════════

    def _wait_until_reached(self, target,
                            tolerance=0.02, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            errors = [
                abs(c - t)
                for c, t in zip(self._cache_joint_pos, target)
            ]
            if all(e < tolerance for e in errors):
                return True
            time.sleep(0.02)
        return False

    def _simple_ik(self, target_pos,
                   max_iter=200, step=0.5, tol=0.005):
        """Простой Якобиан-IK до точки target_pos [x, y, z]."""
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        if site_id < 0:
            logger.add("[MuJoCo] IK: сайт 'tcp' не найден")
            return False

        # DOF-адреса суставов из конфига
        joint_dof_ids = []
        for jname in self._mujoco_joint_names:
            jid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                logger.add(f"[MuJoCo] IK: сустав '{jname}' не найден")
                return False
            joint_dof_ids.append(self._model.jnt_dofadr[jid])

        for _ in range(max_iter):
            current_pos = np.array(self._cache_tcp_pos)
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                return True

            jac_pos = np.zeros((3, self._model.nv))
            jac_rot = np.zeros((3, self._model.nv))
            with self._lock:
                mujoco.mj_jacSite(
                    self._model, self._data,
                    jac_pos, jac_rot, site_id)

            J = jac_pos[:, joint_dof_ids]
            dq = step * np.linalg.pinv(J) @ error

            for i, di in enumerate(dq):
                self._target_joints[i] += float(di)
            time.sleep(0.02)

        return False

    @staticmethod
    def _quat_to_euler(quat):
        """Кватернион (w, x, y, z) → углы Эйлера (rx, ry, rz)."""
        w, x, y, z = quat
        rx = np.arctan2(2 * (w * x + y * z),
                        1 - 2 * (x * x + y * y))
        sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
        ry = np.arcsin(sinp)
        rz = np.arctan2(2 * (w * z + x * y),
                        1 - 2 * (y * y + z * z))
        return np.array([rx, ry, rz])