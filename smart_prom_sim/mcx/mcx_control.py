# smart_prom_sim/mcx/mcx_control.py
import time
import numpy as np
from typing import List, Tuple

class MCX:
    """
    Заглушка для реального робота MCX
    Позволяет полностью запускать GUI в оффлайн-режиме
    """
    def __init__(self):
        self.connected = True
        self.current_cart = [0.5, 0.0, 0.5, np.pi/2, 0.0, np.pi]  # X, Y, Z, Rx, Ry, Rz
        self.current_joints = [0.0, 0.0, np.pi/2, 0.0, np.pi/2, 0.0]
        self._gripper_state = 0
        print("[MOCK] Робот MCX инициализирован (заглушка)")

    def connect(self, ip: str = "192.168.2.100"):
        print(f"[MOCK] Подключение к {ip} — УСПЕШНО (симуляция)")
        self.connected = True
        return True

    def get_connection(self) -> bool:
        return self.connected

    def move_to_start(self):
        print("[MOCK] Возврат в стартовую позицию")
        self.current_joints = [0.0, 0.0, np.pi/2, 0.0, np.pi/2, 0.0]
        self.current_cart = [0.5, 0.0, 0.5, np.pi/2, 0.0, np.pi]
        time.sleep(0.5)

    def get_cart_pos(self) -> List[float]:
        return self.current_cart.copy()

    def get_joint_pos(self) -> List[float]:
        return self.current_joints.copy()

    def MoveJ(self, joints: List[float]):
        if len(joints) != 6:
            print(f"[MOCK] ОШИБКА: MoveJ ожидает 6 значений, получено {len(joints)}")
            return
        print(f"[MOCK] MoveJ → {np.round(joints, 3)}")
        self.current_joints = joints.copy()
        # Имитация движения
        time.sleep(0.3)

    def MoveL(self, pose: List[float]):
        if len(pose) != 6:
            print(f"[MOCK] ОШИБКА: MoveL ожидает 6 значений, получено {len(pose)}")
            return
        print(f"[MOCK] MoveL → {np.round(pose, 3)}")
        self.current_cart = pose.copy()
        time.sleep(0.3)

    @property
    def gripper_state(self):
        return self._gripper_state

    @gripper_state.setter
    def gripper_state(self, state: int):
        state = 1 if state else 0
        if self._gripper_state != state:
            self._gripper_state = state
            action = "ОТКРЫТ" if state else "ЗАКРЫТ"
            print(f"[MOCK] Схват {action}")

    # Совместимость с реальным API
    def get_actual_temperature(self):
        return [25 + i*3 for i in range(6)]  # имитация температуры

    def emergency_stop(self):
        print("[MOCK] ЭКСТРЕННАЯ ОСТАНОВКА!")
        self.current_cart = self.current_cart[:3] + [np.pi/2, 0, np.pi]
        time.sleep(0.1)