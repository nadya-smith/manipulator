# robot_backend/base_robot.py

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from enum import Enum


class RobotMode(Enum):
    """Режим работы робота."""
    REAL = "real"
    SIMULATION = "simulation"


class RobotState(Enum):
    """Состояние робота."""
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    MOVING = "moving"
    PAUSED = "paused"
    EMERGENCY = "emergency"
    ERROR = "error"


class BaseRobot(ABC):
    """
    Абстрактный интерфейс робота.
    
    Все реализации (реальный робот, MuJoCo симуляция)
    должны наследовать этот класс и реализовать все абстрактные методы.
    API остаётся одинаковым — переключение прозрачно для вызывающего кода.
    """

    def __init__(self):
        self._state: RobotState = RobotState.DISCONNECTED
        self._mode: RobotMode = RobotMode.REAL
        self._num_joints: int = 6
        self._gripper_state: float = 0.0  # 0.0 = закрыт, 1.0 = открыт

    # ═══════════════════════════════════════════════════
    #  Свойства (read-only)
    # ═══════════════════════════════════════════════════

    @property
    def state(self) -> RobotState:
        return self._state

    @property
    def mode(self) -> RobotMode:
        return self._mode

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def gripper_state(self) -> float:
        return self._gripper_state

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Подключён ли робот / инициализирована ли симуляция."""
        ...

    @property
    @abstractmethod
    def is_moving(self) -> bool:
        """Находится ли робот в движении."""
        ...

    # ═══════════════════════════════════════════════════
    #  Подключение / отключение
    # ═══════════════════════════════════════════════════

    @abstractmethod
    def connect(self, target: str = "", **kwargs) -> bool:
        """
        Подключение к роботу или инициализация симуляции.
        
        Args:
            target: для реального робота — COM-порт ("COM39"),
                    для MuJoCo — путь к XML-модели.
            **kwargs: дополнительные параметры (baudrate, и т.д.)
        
        Returns:
            True если подключение успешно.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Отключение от робота / остановка симуляции."""
        ...

    # ═══════════════════════════════════════════════════
    #  Чтение состояния
    # ═══════════════════════════════════════════════════

    @abstractmethod
    def get_joint_positions(self) -> List[float]:
        """
        Текущие углы суставов (в радианах).
        
        Returns:
            Список из num_joints значений [j1, j2, ..., j6].
        """
        ...

    @abstractmethod
    def get_joint_velocities(self) -> List[float]:
        """
        Текущие скорости суставов (рад/с).
        
        Returns:
            Список из num_joints значений.
        """
        ...

    @abstractmethod
    def get_cartesian_pose(self) -> List[float]:
        """
        Текущая декартова поза инструмента.
        
        Returns:
            [X, Y, Z, Rx, Ry, Rz] — позиция (м) + ориентация (рад).
        """
        ...

    @abstractmethod
    def get_joint_torques(self) -> List[float]:
        """
        Текущие крутящие моменты на суставах (Н·м).
        
        Returns:
            Список из num_joints значений.
        """
        ...

    # ═══════════════════════════════════════════════════
    #  Управление движением
    # ═══════════════════════════════════════════════════

    @abstractmethod
    def move_j(self, joint_positions: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        """
        Движение по суставам (joint space).
        
        Args:
            joint_positions: целевые углы [j1..j6] в радианах.
            speed: относительная скорость (0.0 — 1.0).
            acceleration: относительное ускорение (0.0 — 1.0).
            blocking: ждать завершения движения.
        
        Returns:
            True если команда принята.
        """
        ...

    @abstractmethod
    def move_l(self, cartesian_pose: List[float],
               speed: float = 1.0, acceleration: float = 1.0,
               blocking: bool = True) -> bool:
        """
        Линейное движение в декартовом пространстве.
        
        Args:
            cartesian_pose: целевая поза [X, Y, Z, Rx, Ry, Rz].
            speed: относительная скорость (0.0 — 1.0).
            acceleration: относительное ускорение (0.0 — 1.0).
            blocking: ждать завершения движения.
        
        Returns:
            True если команда принята.
        """
        ...

    @abstractmethod
    def move_to_home(self) -> bool:
        """Перемещение в домашнюю позицию."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Плавная остановка текущего движения."""
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """Экстренная остановка (мгновенная)."""
        ...

    @abstractmethod
    def pause(self) -> None:
        """Приостановка движения (с возможностью возобновления)."""
        ...

    @abstractmethod
    def resume(self) -> None:
        """Возобновление приостановленного движения."""
        ...

    # ═══════════════════════════════════════════════════
    #  Управление схватом
    # ═══════════════════════════════════════════════════

    @abstractmethod
    def set_gripper(self, value: float) -> None:
        """
        Установка положения схвата.
        
        Args:
            value: 0.0 = полностью закрыт, 1.0 = полностью открыт.
        """
        ...

    @abstractmethod
    def get_gripper(self) -> float:
        """Текущее положение схвата (0.0 — 1.0)."""
        ...

    # ═══════════════════════════════════════════════════
    #  Дополнительные возможности
    # ═══════════════════════════════════════════════════

    @abstractmethod
    def get_camera_frame(self) -> Optional['np.ndarray']:
        """
        Получение кадра с камеры (реальной или виртуальной).
        
        Returns:
            BGR-кадр (numpy array) или None если камера недоступна.
        """
        ...

    @abstractmethod
    def get_info(self) -> dict:
        """
        Словарь с общей информацией о роботе / симуляции.
        
        Returns:
            {"mode": ..., "state": ..., "model": ..., ...}
        """
        ...

    # ═══════════════════════════════════════════════════
    #  Утилитарные методы (общие)
    # ═══════════════════════════════════════════════════

    def is_ready(self) -> bool:
        """Готов ли робот принимать команды."""
        return self._state in (RobotState.IDLE,)

    def __repr__(self):
        return (f"<{self.__class__.__name__} "
                f"mode={self._mode.value} state={self._state.value}>")