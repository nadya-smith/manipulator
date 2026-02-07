# robot_backend/real_robot.py

import numpy as np
from typing import List, Optional
from .base_robot import BaseRobot, RobotMode, RobotState
from logger import logger


class RealRobot(BaseRobot):
    """
    Реализация интерфейса BaseRobot для реального робота MCX.
    Оборачивает существующий mcx_wrapper.RobotMCX.
    """

    def __init__(self):
        super().__init__()
        self._mode = RobotMode.REAL
        self._mcx = None          # экземпляр RobotMCX
        self._connected = False
        self._paused = False

    # ── Свойства ─────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        if self._mcx is None:
            return False
        try:
            return self._mcx.robot.get_connection()
        except Exception:
            return False

    @property
    def is_moving(self) -> bool:
        # MCX API: проверяем, есть ли активное движение
        try:
            if self._mcx:
                return self._mcx.robot.is_in_motion()
        except AttributeError:
            pass
        return False

    # ── Подключение ──────────────────────────────────────

    def connect(self, target: str = "COM39", **kwargs) -> bool:
        try:
            from mcx_wrapper import RobotMCX
            self._mcx = RobotMCX()
            self._mcx.connect(target)
            self._connected = True
            self._state = RobotState.IDLE
            logger.add(f"[РеальныйРобот] Подключён к {target}")
            return True
        except Exception as e:
            logger.add(f"[РеальныйРобот] Ошибка подключения: {e}")
            self._state = RobotState.ERROR
            return False

    def disconnect(self) -> None:
        if self._mcx:
            try:
                self._mcx.move_to_start()
            except Exception:
                pass
            self._mcx = None
        self._connected = False
        self._state = RobotState.DISCONNECTED
        logger.add("[РеальныйРобот] Отключён")

    # ── Чтение состояния ─────────────────────────────────

    def get_joint_positions(self) -> List[float]:
        if not self._mcx:
            return [0.0] * self._num_joints
        try:
            return list(self._mcx.get_joint_pos())
        except Exception as e:
            logger.add(f"[РеальныйРобот] Ошибка чтения суставов: {e}")
            return [0.0] * self._num_joints

    def get_joint_velocities(self) -> List[float]:
        # MCX может не поддерживать — возвращаем нули
        return [0.0] * self._num_joints

    def get_cartesian_pose(self) -> List[float]:
        if not self._mcx:
            return [0.0] * 6
        try:
            return list(self._mcx.get_cart_pos())
        except Exception as e:
            logger.add(f"[РеальныйРобот] Ошибка чтения позы: {e}")
            return [0.0] * 6

    def get_joint_torques(self) -> List[float]:
        return [0.0] * self._num_joints

    # ── Управление движением ─────────────────────────────

    def move_j(self, joint_positions: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        if not self._mcx or self._state == RobotState.EMERGENCY:
            return False
        try:
            self._state = RobotState.MOVING
            self._mcx.move_j(joint_positions)
            self._state = RobotState.IDLE
            return True
        except Exception as e:
            logger.add(f"[РеальныйРобот] Ошибка MoveJ: {e}")
            self._state = RobotState.ERROR
            return False

    def move_l(self, cartesian_pose: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        if not self._mcx or self._state == RobotState.EMERGENCY:
            return False
        try:
            self._state = RobotState.MOVING
            self._mcx.move_l(cartesian_pose)
            self._state = RobotState.IDLE
            return True
        except Exception as e:
            logger.add(f"[РеальныйРобот] Ошибка MoveL: {e}")
            self._state = RobotState.ERROR
            return False

    def move_to_home(self) -> bool:
        if not self._mcx:
            return False
        try:
            self._mcx.move_to_start()
            return True
        except Exception as e:
            logger.add(f"[РеальныйРобот] Ошибка Home: {e}")
            return False

    def stop(self) -> None:
        logger.add("[РеальныйРобот] Остановка")
        self._state = RobotState.IDLE

    def emergency_stop(self) -> None:
        if self._mcx:
            try:
                self._mcx.emergency_stop()
            except Exception:
                pass
        self._state = RobotState.EMERGENCY
        logger.add("[РеальныйРобот] ЭКСТРЕННАЯ ОСТАНОВКА")

    def pause(self) -> None:
        self._paused = True
        self._state = RobotState.PAUSED
        logger.add("[РеальныйРобот] Пауза")

    def resume(self) -> None:
        if self._paused:
            self._paused = False
            self._state = RobotState.IDLE
            logger.add("[РеальныйРобот] Возобновление")

    # ── Схват ─────────────────────────────────────────────

    def set_gripper(self, value: float) -> None:
        if self._mcx:
            try:
                self._mcx.set_gripper(int(round(value)))
                self._gripper_state = value
            except Exception as e:
                logger.add(f"[РеальныйРобот] Ошибка схвата: {e}")

    def get_gripper(self) -> float:
        return self._gripper_state

    # ── Камера ─────────────────────────────────────────────

    def get_camera_frame(self) -> Optional[np.ndarray]:
        # Реальная камера обрабатывается через CameraThread
        return None

    # ── Инфо ──────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "mode": self._mode.value,
            "state": self._state.value,
            "model": "MCX Collaborative Robot",
            "connected": self.is_connected,
            "num_joints": self._num_joints,
        }