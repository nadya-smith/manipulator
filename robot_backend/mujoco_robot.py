# robot_backend/mujoco_robot.py

import numpy as np
import threading
import time
from typing import List, Optional
from .base_robot import BaseRobot, RobotMode, RobotState
from logger import logger

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.add("[MuJoCo] Библиотека mujoco не установлена. pip install mujoco")


class MuJoCoRobot(BaseRobot):
    """
    Реализация интерфейса BaseRobot через MuJoCo симуляцию.
    Полностью совместима по API с RealRobot.
    """

    # Домашнее положение (рад)
    HOME_POSITION = [0.0, -1.5708, 1.5708, 0.0, 1.5708, 0.0]

    # MJCF-модель по умолчанию (6-DOF робот-манипулятор)
    DEFAULT_MODEL_XML = """
    <mujoco model="6dof_robot">
      <compiler angle="radian"/>

      <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit"/>

      <default>
        <joint damping="5" armature="0.1"/>
        <geom rgba="0.8 0.8 0.8 1" condim="3" friction="1 0.5 0.001"/>
        <position kp="100" kv="10"/>
      </default>

      <worldbody>
        <!-- Пол -->
        <geom type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1"/>

        <!-- Подсветка -->
        <light diffuse="0.8 0.8 0.8" pos="0 0 4" dir="0 0 -1"/>

        <!-- Основание робота -->
        <body name="base" pos="0 0 0.05">
          <geom type="cylinder" size="0.12 0.05" rgba="0.3 0.3 0.3 1"/>

          <!-- Звено 1 (вращение вокруг Z) -->
          <body name="link1" pos="0 0 0.05">
            <joint name="joint1" type="hinge" axis="0 0 1"
                   range="-3.14159 3.14159"/>
            <geom type="cylinder" size="0.06 0.15" pos="0 0 0.15"
                  rgba="0.2 0.4 0.8 1"/>

            <!-- Звено 2 -->
            <body name="link2" pos="0 0 0.3">
              <joint name="joint2" type="hinge" axis="0 1 0"
                     range="-2.356 2.356"/>
              <geom type="capsule" size="0.05" fromto="0 0 0  0 0 0.3"
                    rgba="0.2 0.6 0.2 1"/>

              <!-- Звено 3 -->
              <body name="link3" pos="0 0 0.3">
                <joint name="joint3" type="hinge" axis="0 1 0"
                       range="-2.356 2.356"/>
                <geom type="capsule" size="0.04" fromto="0 0 0  0 0 0.25"
                      rgba="0.8 0.4 0.2 1"/>

                <!-- Звено 4 -->
                <body name="link4" pos="0 0 0.25">
                  <joint name="joint4" type="hinge" axis="0 0 1"
                         range="-3.14159 3.14159"/>
                  <geom type="cylinder" size="0.035 0.04" pos="0 0 0.04"
                        rgba="0.6 0.2 0.6 1"/>

                  <!-- Звено 5 -->
                  <body name="link5" pos="0 0 0.08">
                    <joint name="joint5" type="hinge" axis="0 1 0"
                           range="-2.356 2.356"/>
                    <geom type="cylinder" size="0.03 0.03" pos="0 0 0.03"
                          rgba="0.8 0.8 0.2 1"/>

                    <!-- Звено 6 (инструмент) -->
                    <body name="link6" pos="0 0 0.06">
                      <joint name="joint6" type="hinge" axis="0 0 1"
                             range="-3.14159 3.14159"/>
                      <geom type="cylinder" size="0.025 0.02" pos="0 0 0.02"
                            rgba="0.9 0.1 0.1 1"/>

                      <!-- TCP (Tool Center Point) -->
                      <site name="tcp" pos="0 0 0.04" size="0.01"
                            rgba="1 0 0 1"/>

                      <!-- Схват: левый палец -->
                      <body name="gripper_left" pos="0 -0.02 0.05">
                        <joint name="gripper_left_joint" type="slide"
                               axis="0 1 0" range="0 0.03"/>
                        <geom type="box" size="0.01 0.005 0.025"
                              rgba="0.5 0.5 0.5 1"/>
                      </body>

                      <!-- Схват: правый палец -->
                      <body name="gripper_right" pos="0 0.02 0.05">
                        <joint name="gripper_right_joint" type="slide"
                               axis="0 -1 0" range="0 0.03"/>
                        <geom type="box" size="0.01 0.005 0.025"
                              rgba="0.5 0.5 0.5 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>

        <!-- ═══ Объекты на сцене (для визуализации / захвата) ═══ -->

        <!-- Красный куб -->
        <body name="red_cube" pos="0.4 0.2 0.025">
          <freejoint/>
          <geom type="box" size="0.025 0.025 0.025" mass="0.1"
                rgba="1 0 0 1"/>
        </body>

        <!-- Зелёный цилиндр -->
        <body name="green_cylinder" pos="0.3 -0.25 0.025">
          <freejoint/>
          <geom type="cylinder" size="0.02 0.025" mass="0.08"
                rgba="0 0.8 0 1"/>
        </body>

        <!-- Синяя сфера -->
        <body name="blue_sphere" pos="-0.3 0.3 0.03">
          <freejoint/>
          <geom type="sphere" size="0.03" mass="0.05"
                rgba="0 0 1 1"/>
        </body>

        <!-- Камера сцены -->
        <camera name="scene_camera" pos="1.2 -0.8 1.0"
                xyaxes="0.6 0.8 0 -0.3 0.2 0.9"/>

      </worldbody>

      <!-- ═══ Актуаторы ═══ -->
      <actuator>
        <position name="act_j1" joint="joint1" ctrlrange="-3.14159 3.14159"
                  kp="200"/>
        <position name="act_j2" joint="joint2" ctrlrange="-2.356 2.356"
                  kp="300"/>
        <position name="act_j3" joint="joint3" ctrlrange="-2.356 2.356"
                  kp="200"/>
        <position name="act_j4" joint="joint4" ctrlrange="-3.14159 3.14159"
                  kp="100"/>
        <position name="act_j5" joint="joint5" ctrlrange="-2.356 2.356"
                  kp="100"/>
        <position name="act_j6" joint="joint6" ctrlrange="-3.14159 3.14159"
                  kp="50"/>
        <position name="act_gripper_l" joint="gripper_left_joint"
                  ctrlrange="0 0.03" kp="50"/>
        <position name="act_gripper_r" joint="gripper_right_joint"
                  ctrlrange="0 0.03" kp="50"/>
      </actuator>

      <!-- ═══ Сенсоры ═══ -->
      <sensor>
        <jointpos name="sens_j1" joint="joint1"/>
        <jointpos name="sens_j2" joint="joint2"/>
        <jointpos name="sens_j3" joint="joint3"/>
        <jointpos name="sens_j4" joint="joint4"/>
        <jointpos name="sens_j5" joint="joint5"/>
        <jointpos name="sens_j6" joint="joint6"/>
        <jointvel name="sensv_j1" joint="joint1"/>
        <jointvel name="sensv_j2" joint="joint2"/>
        <jointvel name="sensv_j3" joint="joint3"/>
        <jointvel name="sensv_j4" joint="joint4"/>
        <jointvel name="sensv_j5" joint="joint5"/>
        <jointvel name="sensv_j6" joint="joint6"/>
        <actuatorfrc name="frc_j1" actuator="act_j1"/>
        <actuatorfrc name="frc_j2" actuator="act_j2"/>
        <actuatorfrc name="frc_j3" actuator="act_j3"/>
        <actuatorfrc name="frc_j4" actuator="act_j4"/>
        <actuatorfrc name="frc_j5" actuator="act_j5"/>
        <actuatorfrc name="frc_j6" actuator="act_j6"/>
        <framepos name="tcp_pos" objtype="site" objname="tcp"/>
        <framequat name="tcp_quat" objtype="site" objname="tcp"/>
      </sensor>
    </mujoco>
    """

    def __init__(self):
        super().__init__()
        self._mode = RobotMode.SIMULATION
        self._model = None       # mujoco.MjModel
        self._data = None        # mujoco.MjData
        self._renderer = None    # для рендеринга кадров
        self._viewer = None      # интерактивный 3D-viewer
        self._viewer_thread = None

        self._sim_thread = None
        self._sim_running = False
        self._sim_lock = threading.Lock()

        self._target_joints = list(self.HOME_POSITION)
        self._target_gripper = 0.0

        self._paused = False
        self._emergency = False

        # Маппинг: имя актуатора → индекс
        self._joint_actuator_ids = []  # [0..5]
        self._gripper_actuator_ids = []  # [6, 7]

        # Маппинг сенсоров
        self._joint_pos_sensor_ids = []  # sens_j1..6
        self._joint_vel_sensor_ids = []  # sensv_j1..6
        self._joint_frc_sensor_ids = []  # frc_j1..6
        self._tcp_pos_sensor_id = -1
        self._tcp_quat_sensor_id = -1

    # ═══════════════════════════════════════════════════
    #  Свойства
    # ═══════════════════════════════════════════════════

    @property
    def is_connected(self) -> bool:
        return self._model is not None and self._data is not None

    @property
    def is_moving(self) -> bool:
        if not self.is_connected:
            return False
        vels = self.get_joint_velocities()
        return any(abs(v) > 0.01 for v in vels)

    # ═══════════════════════════════════════════════════
    #  Подключение (= инициализация симуляции)
    # ═══════════════════════════════════════════════════

    def connect(self, target: str = "", **kwargs) -> bool:
        """
        Инициализация MuJoCo симуляции.
        
        Args:
            target: путь к XML-файлу модели. Пусто = встроенная модель.
            **kwargs:
                show_viewer (bool): запустить интерактивный 3D-просмотрщик.
                render_width (int): ширина оффскрин-рендера.
                render_height (int): высота оффскрин-рендера.
        """
        if not MUJOCO_AVAILABLE:
            logger.add("[MuJoCo] Библиотека не установлена!")
            self._state = RobotState.ERROR
            return False

        try:
            # Загрузка модели
            if target and target.endswith(".xml"):
                self._model = mujoco.MjModel.from_xml_path(target)
                logger.add(f"[MuJoCo] Загружена модель из {target}")
            else:
                self._model = mujoco.MjModel.from_xml_string(
                    self.DEFAULT_MODEL_XML)
                logger.add("[MuJoCo] Загружена встроенная модель")

            self._data = mujoco.MjData(self._model)

            # Определяем индексы актуаторов
            self._resolve_ids()

            # Устанавливаем домашнюю позицию
            for i, jid in enumerate(self._joint_actuator_ids):
                self._data.ctrl[jid] = self.HOME_POSITION[i]
            mujoco.mj_step(self._model, self._data)

            # Инициализация рендерера
            rw = kwargs.get("render_width", 640)
            rh = kwargs.get("render_height", 480)
            self._renderer = mujoco.Renderer(self._model, height=rh, width=rw)

            # Запуск потока симуляции
            self._sim_running = True
            self._emergency = False
            self._paused = False
            self._sim_thread = threading.Thread(
                target=self._simulation_loop, daemon=True)
            self._sim_thread.start()

            # 3D-просмотрщик (опционально)
            if kwargs.get("show_viewer", False):
                self._start_viewer()

            self._state = RobotState.IDLE
            logger.add("[MuJoCo] Симуляция запущена")
            return True

        except Exception as e:
            logger.add(f"[MuJoCo] Ошибка инициализации: {e}")
            self._state = RobotState.ERROR
            return False

    def disconnect(self) -> None:
        self._sim_running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=2.0)
            self._sim_thread = None

        if self._viewer:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

        self._renderer = None
        self._model = None
        self._data = None
        self._state = RobotState.DISCONNECTED
        logger.add("[MuJoCo] Симуляция остановлена")

    # ── Поиск ID актуаторов / сенсоров ────────────────

    def _resolve_ids(self):
        """Маппинг имён актуаторов и сенсоров → числовые ID."""
        self._joint_actuator_ids = []
        for i in range(1, 7):
            name = f"act_j{i}"
            aid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self._joint_actuator_ids.append(aid)

        self._gripper_actuator_ids = []
        for name in ["act_gripper_l", "act_gripper_r"]:
            aid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self._gripper_actuator_ids.append(aid)

        self._joint_pos_sensor_ids = []
        self._joint_vel_sensor_ids = []
        self._joint_frc_sensor_ids = []
        for i in range(1, 7):
            self._joint_pos_sensor_ids.append(
                mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"sens_j{i}"))
            self._joint_vel_sensor_ids.append(
                mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensv_j{i}"))
            self._joint_frc_sensor_ids.append(
                mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"frc_j{i}"))

        self._tcp_pos_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "tcp_pos")
        self._tcp_quat_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "tcp_quat")

    # ═══════════════════════════════════════════════════
    #  Цикл симуляции
    # ═══════════════════════════════════════════════════

    def _simulation_loop(self):
        """Основной цикл физической симуляции (отдельный поток)."""
        dt = self._model.opt.timestep
        logger.add(f"[MuJoCo] Цикл симуляции, dt={dt:.4f}с")

        while self._sim_running:
            if self._paused or self._emergency:
                time.sleep(0.01)
                continue

            with self._sim_lock:
                # Применяем управляющие сигналы на актуаторы суставов
                for i, aid in enumerate(self._joint_actuator_ids):
                    self._data.ctrl[aid] = self._target_joints[i]

                # Схват
                gripper_val = self._target_gripper * 0.03  # нормализация
                for gid in self._gripper_actuator_ids:
                    self._data.ctrl[gid] = gripper_val

                # Шаг симуляции
                mujoco.mj_step(self._model, self._data)

            # Поддерживаем реальное время
            time.sleep(dt)

    def _start_viewer(self):
        """Запуск интерактивного 3D-просмотрщика в отдельном потоке."""
        def _viewer_fn():
            try:
                with mujoco.viewer.launch_passive(
                        self._model, self._data) as v:
                    self._viewer = v
                    while v.is_running() and self._sim_running:
                        v.sync()
                        time.sleep(0.016)  # ~60 FPS
            except Exception as e:
                logger.add(f"[MuJoCo] Ошибка 3D-просмотрщика: {e}")

        self._viewer_thread = threading.Thread(target=_viewer_fn, daemon=True)
        self._viewer_thread.start()
        logger.add("[MuJoCo] 3D-просмотрщик запущен")

    # ═══════════════════════════════════════════════════
    #  Чтение состояния
    # ═══════════════════════════════════════════════════

    def get_joint_positions(self) -> List[float]:
        if not self.is_connected:
            return [0.0] * self._num_joints
        with self._sim_lock:
            result = []
            for sid in self._joint_pos_sensor_ids:
                adr = self._model.sensor_adr[sid]
                result.append(float(self._data.sensordata[adr]))
            return result

    def get_joint_velocities(self) -> List[float]:
        if not self.is_connected:
            return [0.0] * self._num_joints
        with self._sim_lock:
            result = []
            for sid in self._joint_vel_sensor_ids:
                adr = self._model.sensor_adr[sid]
                result.append(float(self._data.sensordata[adr]))
            return result

    def get_cartesian_pose(self) -> List[float]:
        """Возвращает [X, Y, Z, Rx, Ry, Rz] TCP."""
        if not self.is_connected:
            return [0.0] * 6
        with self._sim_lock:
            # Позиция TCP
            pos_adr = self._model.sensor_adr[self._tcp_pos_sensor_id]
            pos = self._data.sensordata[pos_adr:pos_adr + 3].copy()

            # Кватернион → Эйлер (RPY)
            quat_adr = self._model.sensor_adr[self._tcp_quat_sensor_id]
            quat = self._data.sensordata[quat_adr:quat_adr + 4].copy()

        rpy = self._quat_to_euler(quat)
        return [float(pos[0]), float(pos[1]), float(pos[2]),
                float(rpy[0]), float(rpy[1]), float(rpy[2])]

    def get_joint_torques(self) -> List[float]:
        if not self.is_connected:
            return [0.0] * self._num_joints
        with self._sim_lock:
            result = []
            for sid in self._joint_frc_sensor_ids:
                adr = self._model.sensor_adr[sid]
                result.append(float(self._data.sensordata[adr]))
            return result

    # ═══════════════════════════════════════════════════
    #  Управление движением
    # ═══════════════════════════════════════════════════

    def move_j(self, joint_positions: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        if not self.is_connected or self._emergency:
            return False

        self._state = RobotState.MOVING
        self._target_joints = list(joint_positions[:self._num_joints])

        if blocking:
            self._wait_until_reached(joint_positions, timeout=10.0)

        self._state = RobotState.IDLE
        logger.add(f"[MuJoCo] MoveJ → "
                    f"{[f'{np.degrees(j):.1f}°' for j in joint_positions]}")
        return True

    def move_l(self, cartesian_pose: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        """
        Линейное движение — используем простую IK (итеративный Якобиан).
        В реальном проекте здесь был бы полноценный IK-солвер.
        """
        if not self.is_connected or self._emergency:
            return False

        self._state = RobotState.MOVING

        # Простая стратегия: решаем IK через Якобиан MuJoCo
        target_pos = np.array(cartesian_pose[:3])
        success = self._simple_ik(target_pos, max_iter=200)

        if not success:
            logger.add("[MuJoCo] MoveL: IK не сошлась")

        self._state = RobotState.IDLE
        logger.add(f"[MuJoCo] MoveL → "
                    f"X={cartesian_pose[0]:.3f} "
                    f"Y={cartesian_pose[1]:.3f} "
                    f"Z={cartesian_pose[2]:.3f}")
        return success

    def move_to_home(self) -> bool:
        return self.move_j(self.HOME_POSITION, blocking=True)

    def stop(self) -> None:
        """Плавная остановка: ставим целевые позиции = текущим."""
        if self.is_connected:
            current = self.get_joint_positions()
            self._target_joints = current
        self._state = RobotState.IDLE
        logger.add("[MuJoCo] Остановка")

    def emergency_stop(self) -> None:
        self._emergency = True
        if self.is_connected:
            current = self.get_joint_positions()
            self._target_joints = current
        self._state = RobotState.EMERGENCY
        logger.add("[MuJoCo] ЭКСТРЕННАЯ ОСТАНОВКА")

    def pause(self) -> None:
        self._paused = True
        self._state = RobotState.PAUSED
        logger.add("[MuJoCo] Пауза")

    def resume(self) -> None:
        self._paused = False
        self._emergency = False
        self._state = RobotState.IDLE
        logger.add("[MuJoCo] Возобновление")

    # ═══════════════════════════════════════════════════
    #  Схват
    # ═══════════════════════════════════════════════════

    def set_gripper(self, value: float) -> None:
        self._target_gripper = max(0.0, min(1.0, value))
        self._gripper_state = self._target_gripper
        state_str = "ОТКРЫТ" if value > 0.5 else "ЗАКРЫТ"
        logger.add(f"[MuJoCo] Схват: {state_str} ({value:.2f})")

    def get_gripper(self) -> float:
        return self._gripper_state

    # ═══════════════════════════════════════════════════
    #  Камера (оффскрин-рендеринг)
    # ═══════════════════════════════════════════════════

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Рендерит кадр из MuJoCo-камеры (BGR для совместимости с OpenCV)."""
        if not self.is_connected or self._renderer is None:
            return None

        try:
            with self._sim_lock:
                self._renderer.update_scene(
                    self._data, camera="scene_camera")
            rgb = self._renderer.render()
            bgr = rgb[:, :, ::-1].copy()  # RGB → BGR
            return bgr
        except Exception as e:
            logger.add(f"[MuJoCo] Ошибка рендеринга: {e}")
            return None

    # ═══════════════════════════════════════════════════
    #  Инфо
    # ═══════════════════════════════════════════════════

    def get_info(self) -> dict:
        info = {
            "mode": self._mode.value,
            "state": self._state.value,
            "model": "MuJoCo 6-DOF Simulation",
            "connected": self.is_connected,
            "num_joints": self._num_joints,
            "sim_time": float(self._data.time) if self._data else 0.0,
            "timestep": float(self._model.opt.timestep) if self._model else 0.0,
        }
        return info

    # ═══════════════════════════════════════════════════
    #  Утилитарные методы
    # ═══════════════════════════════════════════════════

    def _wait_until_reached(self, target: List[float],
                            tolerance: float = 0.02,
                            timeout: float = 10.0):
        """Блокирующее ожидание достижения целевого положения."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            current = self.get_joint_positions()
            errors = [abs(c - t) for c, t in zip(current, target)]
            if all(e < tolerance for e in errors):
                return True
            time.sleep(0.02)
        logger.add("[MuJoCo] Таймаут ожидания достижения цели")
        return False

    def _simple_ik(self, target_pos: np.ndarray,
                   max_iter: int = 200, step: float = 0.5,
                   tol: float = 0.005) -> bool:
        """
        Простейший IK через итеративный Якобиан MuJoCo.
        Для продакшена использовать IKFast / trac_ik / pinocchio.
        """
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        for iteration in range(max_iter):
            with self._sim_lock:
                current_pos = self._data.site_xpos[site_id].copy()

            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                return True

            # Вычисляем Якобиан
            jac_pos = np.zeros((3, self._model.nv))
            jac_rot = np.zeros((3, self._model.nv))
            with self._sim_lock:
                mujoco.mj_jacSite(
                    self._model, self._data, jac_pos, jac_rot, site_id)

            # Берём только столбцы для наших 6 суставов
            joint_ids = []
            for i in range(1, 7):
                jid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                dof_adr = self._model.jnt_dofadr[jid]
                joint_ids.append(dof_adr)

            J = jac_pos[:, joint_ids]  # (3, 6)

            # Псевдообратный Якобиан
            dq = step * np.linalg.pinv(J) @ error

            # Обновляем целевые углы
            for i, di in enumerate(dq):
                self._target_joints[i] += di

            time.sleep(0.01)

        return False

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """Кватернион [w, x, y, z] → Эйлер [Rx, Ry, Rz] (рад)."""
        w, x, y, z = quat

        # Roll (Rx)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        rx = np.arctan2(sinr, cosr)

        # Pitch (Ry)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        ry = np.arcsin(sinp)

        # Yaw (Rz)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        rz = np.arctan2(siny, cosy)

        return np.array([rx, ry, rz])