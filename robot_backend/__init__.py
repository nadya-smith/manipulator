# robot_backend/__init__.py

from .base_robot import BaseRobot, RobotMode, RobotState
from .real_robot import RealRobot
from .robot_factory import RobotFactory

# MuJoCo — опциональный
try:
    from .mujoco_robot import MuJoCoRobot, MUJOCO_AVAILABLE
except (ImportError, OSError):
    MuJoCoRobot = None
    MUJOCO_AVAILABLE = False

__all__ = [
    "BaseRobot",
    "RobotMode",
    "RobotState",
    "RealRobot",
    "MuJoCoRobot",
    "RobotFactory",
    "MUJOCO_AVAILABLE",
]