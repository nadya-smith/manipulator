from smart_prom_sim.mcx.mcx_control import MCX
from logger import logger
import time

class RobotMCX:
    def __init__(self):
        self.robot = MCX()
        self.connected = False

    def connect(self, ip: str = "192.168.2.100"):
        try:
            self.robot.connect(ip)
            self.connected = self.robot.get_connection()
            if self.connected:
                logger.add("Подключение к роботу установлено")
                self.move_to_start()
            else:
                logger.add("Ошибка подключения к роботу!", emergency=True)
        except Exception as e:
            logger.add(f"Исключение при подключении: {e}", emergency=True)

    def move_to_start(self):
        if self.connected:
            logger.add("Возврат в стартовую позицию")
            self.robot.move_to_start()
            time.sleep(1)

    def get_cart_pos(self):
        return self.robot.get_cart_pos() if self.connected else [0]*6

    def get_joint_pos(self):
        return self.robot.get_joint_pos() if self.connected else [0]*6

    def move_j(self, joints):
        if self.connected:
            logger.add(f"MoveJ → {joints}")
            self.robot.MoveJ(joints)

    def move_l(self, pose):
        if self.connected:
            logger.add(f"MoveL → {pose}")
            self.robot.MoveL(pose)

    def set_gripper(self, state: int):
        if self.connected:
            self.robot.gripper_state = state
            logger.add(f"Схват {'открыт' if state else 'закрыт'}")

    def emergency_stop(self):
        if self.connected:
            logger.add("ЭКСТРЕННАЯ ОСТАНОВКА!", emergency=True)
            # Реальной команды нет — имитируем
            # В реальном API может быть robot.emergency_stop()