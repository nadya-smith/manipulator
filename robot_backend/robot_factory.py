# robot_backend/robot_factory.py

from .base_robot import BaseRobot, RobotMode
from .real_robot import RealRobot
from .mujoco_robot import MuJoCoRobot
from logger import logger


class RobotFactory:
    """
    Фабрика для создания экземпляра робота.
    Позволяет прозрачно переключаться между реальным и виртуальным роботом.
    """

    _registry = {
        RobotMode.REAL: RealRobot,
        RobotMode.SIMULATION: MuJoCoRobot,
    }

    @classmethod
    def create(cls, mode: RobotMode, **kwargs) -> BaseRobot:
        """
        Создаёт экземпляр робота нужного типа.
        
        Args:
            mode: RobotMode.REAL или RobotMode.SIMULATION
            **kwargs: передаются в конструктор
        
        Returns:
            Экземпляр BaseRobot
        """
        robot_class = cls._registry.get(mode)
        if robot_class is None:
            raise ValueError(f"Неизвестный режим: {mode}")

        robot = robot_class(**kwargs)
        logger.add(f"[Фабрика] Создан робот: {robot}")
        return robot

    @classmethod
    def register(cls, mode: RobotMode, robot_class):
        """Регистрация пользовательского бэкенда."""
        cls._registry[mode] = robot_class
        logger.add(f"[Фабрика] Зарегистрирован бэкенд: "
                    f"{mode.value} → {robot_class.__name__}")