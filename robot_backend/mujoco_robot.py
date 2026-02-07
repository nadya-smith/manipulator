# robot_backend/mujoco_robot.py

import numpy as np
import threading
import time
import copy
from typing import List, Optional
from .base_robot import BaseRobot, RobotMode, RobotState
from logger import logger

MUJOCO_AVAILABLE = False

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    logger.add("[MuJoCo] pip install mujoco")
except OSError as e:
    logger.add(f"[MuJoCo] Ошибка DLL: {e}")


class MuJoCoRobot(BaseRobot):

    HOME_POSITION = [0.0, -1.5708, 1.5708, 0.0, 1.5708, 0.0]

    DEFAULT_MODEL_XML = """
    <mujoco model="6dof_robot">
      <compiler angle="radian"/>
      <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit"/>

      <default>
        <joint damping="5" armature="0.1"/>
        <geom rgba="0.8 0.8 0.8 1" condim="3" friction="1 0.5 0.001"/>
      </default>

      <worldbody>
        <geom type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1"/>
        <light diffuse="0.8 0.8 0.8" pos="0 0 4" dir="0 0 -1"/>

        <body name="base" pos="0 0 0.05">
          <geom type="cylinder" size="0.12 0.05" rgba="0.3 0.3 0.3 1"/>
          <body name="link1" pos="0 0 0.05">
            <joint name="joint1" type="hinge" axis="0 0 1"
                   range="-3.14159 3.14159"/>
            <geom type="cylinder" size="0.06 0.15" pos="0 0 0.15"
                  rgba="0.2 0.4 0.8 1"/>
            <body name="link2" pos="0 0 0.3">
              <joint name="joint2" type="hinge" axis="0 1 0"
                     range="-2.356 2.356"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"
                    rgba="0.2 0.6 0.2 1"/>
              <body name="link3" pos="0 0 0.3">
                <joint name="joint3" type="hinge" axis="0 1 0"
                       range="-2.356 2.356"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.25"
                      rgba="0.8 0.4 0.2 1"/>
                <body name="link4" pos="0 0 0.25">
                  <joint name="joint4" type="hinge" axis="0 0 1"
                         range="-3.14159 3.14159"/>
                  <geom type="cylinder" size="0.035 0.04" pos="0 0 0.04"
                        rgba="0.6 0.2 0.6 1"/>
                  <body name="link5" pos="0 0 0.08">
                    <joint name="joint5" type="hinge" axis="0 1 0"
                           range="-2.356 2.356"/>
                    <geom type="cylinder" size="0.03 0.03" pos="0 0 0.03"
                          rgba="0.8 0.8 0.2 1"/>
                    <body name="link6" pos="0 0 0.06">
                      <joint name="joint6" type="hinge" axis="0 0 1"
                             range="-3.14159 3.14159"/>
                      <geom type="cylinder" size="0.025 0.02" pos="0 0 0.02"
                            rgba="0.9 0.1 0.1 1"/>
                      <site name="tcp" pos="0 0 0.04" size="0.01"
                            rgba="1 0 0 1"/>
                      <body name="gripper_left" pos="0 -0.02 0.05">
                        <joint name="gripper_left_joint" type="slide"
                               axis="0 1 0" range="0 0.03"/>
                        <geom type="box" size="0.01 0.005 0.025"
                              rgba="0.5 0.5 0.5 1"/>
                      </body>
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

        <body name="red_cube" pos="0.4 0.2 0.025">
          <freejoint/>
          <geom type="box" size="0.025 0.025 0.025" mass="0.1"
                rgba="1 0 0 1"/>
        </body>
        <body name="green_cylinder" pos="0.3 -0.25 0.025">
          <freejoint/>
          <geom type="cylinder" size="0.02 0.025" mass="0.08"
                rgba="0 0.8 0 1"/>
        </body>
        <body name="blue_sphere" pos="-0.3 0.3 0.03">
          <freejoint/>
          <geom type="sphere" size="0.03" mass="0.05"
                rgba="0 0 1 1"/>
        </body>

        <camera name="scene_camera" pos="1.2 -0.8 1.0"
                xyaxes="0.6 0.8 0 -0.3 0.2 0.9"/>
      </worldbody>

      <actuator>
        <position name="act_j1" joint="joint1"
                  ctrlrange="-3.14159 3.14159" kp="200"/>
        <position name="act_j2" joint="joint2"
                  ctrlrange="-2.356 2.356" kp="300"/>
        <position name="act_j3" joint="joint3"
                  ctrlrange="-2.356 2.356" kp="200"/>
        <position name="act_j4" joint="joint4"
                  ctrlrange="-3.14159 3.14159" kp="100"/>
        <position name="act_j5" joint="joint5"
                  ctrlrange="-2.356 2.356" kp="100"/>
        <position name="act_j6" joint="joint6"
                  ctrlrange="-3.14159 3.14159" kp="50"/>
        <position name="act_gripper_l" joint="gripper_left_joint"
                  ctrlrange="0 0.03" kp="50"/>
        <position name="act_gripper_r" joint="gripper_right_joint"
                  ctrlrange="0 0.03" kp="50"/>
      </actuator>

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
        self._model = None
        self._data = None
        self._renderer = None
        self._viewer = None
        self._viewer_thread = None
        self._sim_thread = None
        self._sim_running = False

        # Главный лок — ВСЕ обращения к self._data через него
        self._lock = threading.Lock()

        # Кэш для чтения из других потоков (обновляется в sim loop)
        self._cache_joint_pos = [0.0] * 6
        self._cache_joint_vel = [0.0] * 6
        self._cache_joint_frc = [0.0] * 6
        self._cache_tcp_pos = [0.0, 0.0, 0.0]
        self._cache_tcp_quat = [1.0, 0.0, 0.0, 0.0]
        self._cache_site_xpos = None

        self._target_joints = list(self.HOME_POSITION)
        self._target_gripper = 0.0
        self._paused = False
        self._emergency = False

        self._joint_actuator_ids = []
        self._gripper_actuator_ids = []
        self._joint_pos_sensor_ids = []
        self._joint_vel_sensor_ids = []
        self._joint_frc_sensor_ids = []
        self._tcp_pos_sensor_id = -1
        self._tcp_quat_sensor_id = -1

        # Флаг: нужно ли рендерить кадр
        self._render_requested = False
        self._rendered_frame = None
        self._render_event = threading.Event()

    # ── Свойства ─────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._model is not None and self._data is not None

    @property
    def is_moving(self) -> bool:
        return any(abs(v) > 0.01 for v in self._cache_joint_vel)

    # ── Подключение ──────────────────────────────────

    def connect(self, target: str = "", **kwargs) -> bool:
        if not MUJOCO_AVAILABLE:
            logger.add("[MuJoCo] Библиотека недоступна!")
            self._state = RobotState.ERROR
            return False

        try:
            if target and target.endswith(".xml"):
                self._model = mujoco.MjModel.from_xml_path(target)
                logger.add(f"[MuJoCo] Модель из {target}")
            else:
                self._model = mujoco.MjModel.from_xml_string(
                    self.DEFAULT_MODEL_XML)
                logger.add("[MuJoCo] Встроенная модель")

            self._data = mujoco.MjData(self._model)
            self._resolve_ids()

            # Начальное положение
            for i, jid in enumerate(self._joint_actuator_ids):
                self._data.ctrl[jid] = self.HOME_POSITION[i]

            # Несколько шагов для стабилизации
            for _ in range(100):
                mujoco.mj_step(self._model, self._data)

            self._update_cache()

            # Рендерер — создаём только если нужно и в том же потоке
            self._renderer = None
            self._render_in_sim_thread = kwargs.get(
                "enable_rendering", True)

            # Запуск потока симуляции
            self._sim_running = True
            self._emergency = False
            self._paused = False
            self._sim_thread = threading.Thread(
                target=self._simulation_loop, daemon=True)
            self._sim_thread.start()

            # Viewer НЕ запускаем автоматически — он конфликтует
            # Используем рендеринг через get_camera_frame()
            if kwargs.get("show_viewer", False):
                logger.add(
                    "[MuJoCo] 3D-viewer отключён для стабильности. "
                    "Используйте виртуальную камеру в GUI.")

            self._state = RobotState.IDLE
            logger.add("[MuJoCo] Симуляция запущена")
            return True

        except Exception as e:
            logger.add(f"[MuJoCo] Ошибка инициализации: {e}")
            import traceback
            logger.add(traceback.format_exc())
            self._state = RobotState.ERROR
            return False

    def disconnect(self) -> None:
        self._sim_running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=3.0)
            self._sim_thread = None
        self._renderer = None
        self._model = None
        self._data = None
        self._state = RobotState.DISCONNECTED
        logger.add("[MuJoCo] Остановлена")

    def _resolve_ids(self):
        self._joint_actuator_ids = []
        for i in range(1, 7):
            aid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_j{i}")
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

    # ── Кэш данных (потокобезопасное чтение) ─────────

    def _update_cache(self):
        """Копирует данные из mj_data в кэш. 
        Вызывается ТОЛЬКО из потока симуляции под локом."""
        sd = self._data.sensordata

        for i, sid in enumerate(self._joint_pos_sensor_ids):
            adr = self._model.sensor_adr[sid]
            self._cache_joint_pos[i] = float(sd[adr])

        for i, sid in enumerate(self._joint_vel_sensor_ids):
            adr = self._model.sensor_adr[sid]
            self._cache_joint_vel[i] = float(sd[adr])

        for i, sid in enumerate(self._joint_frc_sensor_ids):
            adr = self._model.sensor_adr[sid]
            self._cache_joint_frc[i] = float(sd[adr])

        pos_adr = self._model.sensor_adr[self._tcp_pos_sensor_id]
        self._cache_tcp_pos = [
            float(sd[pos_adr]),
            float(sd[pos_adr + 1]),
            float(sd[pos_adr + 2]),
        ]

        quat_adr = self._model.sensor_adr[self._tcp_quat_sensor_id]
        self._cache_tcp_quat = [
            float(sd[quat_adr]),
            float(sd[quat_adr + 1]),
            float(sd[quat_adr + 2]),
            float(sd[quat_adr + 3]),
        ]

    # ── Цикл симуляции (единственный поток с доступом к data) ──

    def _simulation_loop(self):
        dt = self._model.opt.timestep
        logger.add(f"[MuJoCo] Цикл, dt={dt:.4f}с")

        # Рендерер создаём ЗДЕСЬ — в потоке симуляции
        if self._render_in_sim_thread:
            try:
                self._renderer = mujoco.Renderer(
                    self._model, height=480, width=640)
                logger.add("[MuJoCo] Рендерер создан")
            except Exception as e:
                logger.add(f"[MuJoCo] Рендерер недоступен: {e}")
                self._renderer = None

        steps_per_update = 5  # Обновляем кэш каждые N шагов
        step_count = 0

        while self._sim_running:
            if self._paused or self._emergency:
                time.sleep(0.01)
                continue

            with self._lock:
                # Управление
                for i, aid in enumerate(self._joint_actuator_ids):
                    self._data.ctrl[aid] = self._target_joints[i]
                gripper_val = self._target_gripper * 0.03
                for gid in self._gripper_actuator_ids:
                    self._data.ctrl[gid] = gripper_val

                # Шаг физики
                mujoco.mj_step(self._model, self._data)
                step_count += 1

                # Обновляем кэш периодически
                if step_count % steps_per_update == 0:
                    self._update_cache()

                # Рендеринг по запросу
                if (self._render_requested and
                        self._renderer is not None):
                    try:
                        self._renderer.update_scene(
                            self._data, camera="scene_camera")
                        rgb = self._renderer.render()
                        self._rendered_frame = rgb[:, :, ::-1].copy()
                    except Exception as e:
                        self._rendered_frame = None
                    self._render_requested = False
                    self._render_event.set()

            time.sleep(dt)

        logger.add("[MuJoCo] Цикл завершён")

    # ── Чтение состояния (из кэша — потокобезопасно) ──

    def get_joint_positions(self) -> List[float]:
        return list(self._cache_joint_pos)

    def get_joint_velocities(self) -> List[float]:
        return list(self._cache_joint_vel)

    def get_cartesian_pose(self) -> List[float]:
        pos = self._cache_tcp_pos
        rpy = self._quat_to_euler(np.array(self._cache_tcp_quat))
        return [pos[0], pos[1], pos[2],
                float(rpy[0]), float(rpy[1]), float(rpy[2])]

    def get_joint_torques(self) -> List[float]:
        return list(self._cache_joint_frc)

    # ── Движение ─────────────────────────────────────

    def move_j(self, joint_positions: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        if not self.is_connected or self._emergency:
            return False
        self._state = RobotState.MOVING
        self._target_joints = list(joint_positions[:self._num_joints])
        if blocking:
            self._wait_until_reached(joint_positions)
        self._state = RobotState.IDLE
        return True

    def move_l(self, cartesian_pose: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        if not self.is_connected or self._emergency:
            return False
        self._state = RobotState.MOVING
        target_pos = np.array(cartesian_pose[:3])
        success = self._simple_ik(target_pos)
        self._state = RobotState.IDLE
        return success

    def move_to_home(self) -> bool:
        return self.move_j(self.HOME_POSITION, blocking=True)

    def stop(self) -> None:
        self._target_joints = list(self._cache_joint_pos)
        self._state = RobotState.IDLE

    def emergency_stop(self) -> None:
        self._emergency = True
        self._target_joints = list(self._cache_joint_pos)
        self._state = RobotState.EMERGENCY
        logger.add("[MuJoCo] ЭКСТРЕННАЯ ОСТАНОВКА")

    def pause(self) -> None:
        self._paused = True
        self._state = RobotState.PAUSED

    def resume(self) -> None:
        self._paused = False
        self._emergency = False
        self._state = RobotState.IDLE

    # ── Схват ────────────────────────────────────────

    def set_gripper(self, value: float) -> None:
        self._target_gripper = max(0.0, min(1.0, value))
        self._gripper_state = self._target_gripper

    def get_gripper(self) -> float:
        return self._gripper_state

    # ── Камера (рендеринг в потоке симуляции) ─────────

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Запрашивает кадр у потока симуляции и ждёт результат."""
        if not self.is_connected or self._renderer is None:
            return None

        self._render_event.clear()
        self._render_requested = True

        # Ждём пока поток симуляции отрендерит (макс 200мс)
        if self._render_event.wait(timeout=0.2):
            return self._rendered_frame
        return None

    # ── Инфо ─────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "mode": self._mode.value,
            "state": self._state.value,
            "model": "MuJoCo 6-DOF",
            "connected": self.is_connected,
            "num_joints": self._num_joints,
            "sim_time": float(self._data.time) if self._data else 0.0,
        }

    # ── Утилиты ──────────────────────────────────────

    def _wait_until_reached(self, target: List[float],
                            tolerance: float = 0.02,
                            timeout: float = 10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            errors = [abs(c - t) for c, t in
                      zip(self._cache_joint_pos, target)]
            if all(e < tolerance for e in errors):
                return True
            time.sleep(0.02)
        return False

    def _simple_ik(self, target_pos: np.ndarray,
                   max_iter: int = 200, step: float = 0.5,
                   tol: float = 0.005) -> bool:
        """IK через Якобиан — выполняется в потоке симуляции."""
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        joint_dof_ids = []
        for i in range(1, 7):
            jid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            joint_dof_ids.append(self._model.jnt_dofadr[jid])

        for _ in range(max_iter):
            # Читаем из кэша
            current_pos = np.array(self._cache_tcp_pos)
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                return True

            # Якобиан — нужен доступ к data
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
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        rx = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = np.clip(2*(w*y - z*x), -1.0, 1.0)
        ry = np.arcsin(sinp)
        rz = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([rx, ry, rz])