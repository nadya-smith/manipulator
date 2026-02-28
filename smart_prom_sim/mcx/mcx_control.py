import time
import numpy as np
from typing import List

try:
    import Rooky2
except ImportError:
    raise ImportError("Не найдена библиотека Rooky2. Установите её из внутреннего репозитория Promobot.")


class MCX:
    """
    Реальная реализация для манипулятора Rooky.
    Добавлена простая Прямая Кинематика (FK) для обновления координат инструмента при MoveJ.
    """

    def __init__(self):
        self.rooky = None
        self.connected = False
        self.side: str = "right"
        self.port: str = "COM39"

        # Начальное состояние
        self.current_cart = [0.5, 0.0, 0.5, np.pi/2, 0.0, np.pi]
        self.current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._gripper_state = 0

        # Ограничения (мин, макс) в градусах
        self.limits_deg = [
            (-25, 133),  # J1
            (0, 83),     # J2
            (-90, 83),   # J3
            (0, 80),     # J4
            (-86, 86),   # J5
            (-20, 31)    # J6
        ]

        # Приблизительные длины звеньев Rooky (в метрах) для расчета кинематики
        # Если есть точные данные из документации, замените их здесь
        self.L1 = 0.16  # Высота основания
        self.L2 = 0.25  # Плечо
        self.L3 = 0.22  # Предплечье
        self.L4 = 0.15  # Запястье до инструмента

    # ==============================================================================
    def connect(self, port: str = "COM39") -> bool:
        self.port = port
        try:
            self.rooky = Rooky2.Rooky(self.port, self.side)
            self.connected = True
            print(f"[ROOKY] Подключено к {self.port} ({self.side})")
            return True
        except Exception as e:
            print(f"[ROOKY] ОШИБКА подключения к {self.port}: {e}")
            self.connected = False
            return False

    def get_connection(self) -> bool:
        return self.connected

    # ==============================================================================
    def move_to_start(self):
        """Возврат в домашнюю позицию"""
        if not self.connected:
            return
        print("[ROOKY] Возврат в стартовую позицию (reset_joints)")
        self.rooky.reset_joints()
        
        # Сбрасываем углы в 0
        self.current_joints = [0.0] * 6
        self._gripper_state = 0
        
        # Обновляем координаты через кинематику, чтобы GUI увидел изменения
        self._update_fk()
        
        time.sleep(1.5)

    # ==============================================================================
    def get_cart_pos(self) -> List[float]:
        return self.current_cart.copy()

    def get_joint_pos(self) -> List[float]:
        return self.current_joints.copy()

    # ==============================================================================
    def MoveJ(self, joints: List[float]):
        """Движение по суставам с обновлением координат"""
        if not self.connected:
            return
        if len(joints) != 6:
            print(f"[ROOKY] ОШИБКА MoveJ: ожидалось 6 значений")
            return

        # 1. Конвертация и Лимиты
        raw_degrees = [np.rad2deg(j) for j in joints]
        valid_degrees = []
        limit_triggered = False

        for i, deg in enumerate(raw_degrees):
            min_lim, max_lim = self.limits_deg[i]
            clamped = max(min_lim, min(deg, max_lim))
            if clamped != deg:
                limit_triggered = True
            valid_degrees.append(clamped)

        if limit_triggered:
            print(f"[ROOKY] Лимит сработал. Коррекция: {np.round(valid_degrees, 1)}")
        else:
            print(f"[ROOKY] MoveJ → {np.round(valid_degrees, 2)}")

        # 2. Отправка команды роботу
        joints_cmd = [
            {"name": f"{self.side}_arm_{i+1}_joint", "degree": deg}
            for i, deg in enumerate(valid_degrees)
        ]
        self.rooky.move_joints(joints_cmd, 2.0)

        # 3. Обновление внутреннего состояния
        # Сохраняем реальные углы (в радианах) после проверки лимитов
        self.current_joints = [np.deg2rad(d) for d in valid_degrees]
        
        # 4. ВАЖНО: Пересчитываем координаты X, Y, Z на основе новых углов
        self._update_fk()

    # ==============================================================================
    def _update_fk(self):
        """
        Простая прямая кинематика для обновления координат X, Y, Z в GUI.
        Использует упрощенную геометрическую модель.
        """
        j = self.current_joints
        # j[0] = Base Pan (поворот вокруг Z)
        # j[1] = Shoulder Lift (плечо)
        # j[2] = Elbow Lift (локтевой сустав)
        # j[3], j[4], j[5] - кисть (влияют на ориентацию, меньше на позицию)

        # Упрощенный расчет позиции (геометрический метод)
        # Проекция руки на плоскость XY (без учета L1)
        # Эффективная длина проекции от плеча до кисти:
        # r = L2 * cos(j1) + L3 * cos(j1 + j2) ... (углы считаются от вертикали или горизонтали)
        
        # Для простоты считаем классическую схему:
        # Угол 2 и 3 в Rooky часто отсчитываются специфично, но для GUI важно видеть изменения.
        
        # Высота (Z)
        # Z = L1 + L2*sin(j1) + L3*sin(j1+j2) ...
        # Здесь j[1] - плечо, j[2] - локоть.
        
        # Примерная формула (может требовать калибровки знаков под конкретную сборку Rooky):
        proj_len = self.L2 * np.cos(j[1]) + self.L3 * np.cos(j[1] + j[2]) + self.L4 * np.cos(j[1] + j[2] + j[3])
        
        z_calc = self.L1 + self.L2 * np.sin(j[1]) + self.L3 * np.sin(j[1] + j[2]) + self.L4 * np.sin(j[1] + j[2] + j[3])
        x_calc = np.cos(j[0]) * proj_len
        y_calc = np.sin(j[0]) * proj_len

        # Ориентацию (Rx, Ry, Rz) просто аппроксимируем или оставляем как есть, 
        # так как расчет углов Эйлера из матрицы вращения сложнее.
        # Для индикации движения передадим последние 3 сустава как прокси ориентации.
        rx_calc = j[3]
        ry_calc = j[4]
        rz_calc = j[5]

        # Обновляем текущие координаты
        self.current_cart = [x_calc, y_calc, z_calc, rx_calc, ry_calc, rz_calc]

    # ==============================================================================
    def MoveL(self, pose: List[float]):
        """Линейное движение (Симуляция)"""
        if not self.connected:
            return
        if len(pose) != 6:
            print(f"[ROOKY] ОШИБКА MoveL: ожидалось 6 значений")
            return

        print(f"[ROOKY] MoveL → {np.round(pose, 3)} (Sim)")
        self.current_cart = pose.copy()
        # В идеале здесь нужен Inverse Kinematics (IK), чтобы обновить self.current_joints,
        # но для MoveL мы обновляем хотя бы декартовы координаты для GUI.
        time.sleep(0.4)

    # ==============================================================================
    @property
    def gripper_state(self):
        return self._gripper_state

    @gripper_state.setter
    def gripper_state(self, state: int):
        state = 0 if state else 1
        if self._gripper_state == state:
            return
        self._gripper_state = state
        if not self.connected:
            return

        degree = 73 if state else 0
        action = "ЗАКРЫТ" if state else "ОТКРЫТ"
        print(f"[ROOKY] Схват {action} → {degree}°")
        self.rooky.move_joints([{"name": f"{self.side}_arm_7_joint", "degree": degree}], 0.8)

    # ==============================================================================
    def get_actual_temperature(self):
        return [25 + i*3 for i in range(6)]

    def emergency_stop(self):
        print("[ROOKY] ЭКСТРЕННАЯ ОСТАНОВКА")
        if self.connected:
            self.rooky.reset_joints()
            # Сброс внутренних состояний
            self.current_joints = [0.0]*6
            self._update_fk()
        time.sleep(0.5)