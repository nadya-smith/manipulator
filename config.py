"""
config.py — Модуль конфигурации приложения Robot Control.

Поддерживает YAML (приоритет) и JSON.
Если файл конфигурации не найден, используются значения по умолчанию.

Использование:
    from config import cfg

    title = cfg.get("application.window_title")
    fps   = cfg.get("mujoco.rendering.fps", 25)
    cfg.set("sensor.port", "COM5")
    cfg.save()
"""

import os
import json
import copy

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  Значения по умолчанию (единственный источник истины)
# ═══════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    # ── Приложение ── (без изменений)
    "application": {
        "window_title": "Управление роботом — Реальный / MuJoCo + AS5600",
        "window_x": 100,
        "window_y": 100,
        "window_width": 1550,
        "window_height": 1050,
        "update_interval_ms": 200,
    },

    # ── Логирование ── (без изменений)
    "logging": {
        "file": "robot_logs.txt",
        "max_entries": 1000,
    },

    # ── Робот ── (обновлён xml_path)
    "robot": {
        "default_mode": "simulation",
        "real": {
            "port": "COM39",
            "baudrate": 115200,
        },
        "simulation": {
            "xml_path": "mujoco/rooky_arm.xml",       # ← путь к модели
        },
        "movement": {
            "default_step": 5,
            "default_mode": "MoveJ",
        },
        "home_position_deg": [0, 0, 0, 0, 0, 0],
    },

    # ── MuJoCo ── (обновлены камеры)
    "mujoco": {
        "rendering": {
            "fps": 25,
            "width": 640,
            "height": 480,
        },
        "cameras": {
            "overview": "overview_cam",
            "work":     "work_cam",
        },
        "timestep": 0.002,
    },

    # ── Суставы ── (ПОЛНОСТЬЮ ПЕРЕПИСАНО под rooky_arm)
    #   mujoco_name     — имя joint в XML
    #   mujoco_actuator — имя position-актуатора в XML
    "joints": [
        {
            "name": "J1", "label": "Base Pitch",
            "mujoco_name": "J1_base_pitch",
            "mujoco_actuator": "motor_J1",
            "min_deg": -80, "max_deg": 80,
            "gui_invert": True,
            "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False},
        },
        {
            "name": "J2", "label": "Shoulder Pitch",
            "mujoco_name": "J2_shoulder_pitch",
            "mujoco_actuator": "motor_J2_pitch",
            "min_deg": -41.5, "max_deg": 41.5,
            "gui_invert": True,                    # ← + = ОТ стола
            "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False},
        },
        {
            "name": "J3", "label": "Shoulder Roll",
            "mujoco_name": "J2_shoulder_roll",
            "mujoco_actuator": "motor_J2_roll",
            "min_deg": -86.5, "max_deg": 86.5,
            "gui_invert": False,
            "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False},
        },
        {
            "name": "J4", "label": "Elbow Pitch",
            "mujoco_name": "J3_elbow_pitch",
            "mujoco_actuator": "motor_J3_pitch",
            "min_deg": -40, "max_deg": 40,
            "gui_invert": True,                    # ← аналогично плечу
            "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False},
        },
        {
            "name": "J5", "label": "Forearm Roll",
            "mujoco_name": "J3_forearm_roll",
            "mujoco_actuator": "motor_J3_roll",
            "min_deg": -86, "max_deg": 86,
            "gui_invert": False,
            "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False},
        },
        {
            "name": "J6", "label": "Wrist Pitch",
            "mujoco_name": "J4_wrist_pitch",
            "mujoco_actuator": "motor_J4_wrist",
            "min_deg": -25.75, "max_deg": 25.75,
            "gui_invert": True,                    # ← аналогично
            "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False},
        },
    ],

    # ── Схват (НОВАЯ секция) ──
    "gripper": {
        "mujoco_joint":    "J5_grip",
        "mujoco_actuator": "motor_J5_grip",
        "type":            "hinge",       # hinge (рад) / slide (м)
        "open_rad":         0.0,          # data.ctrl при открытом схвате
        "close_rad":        1.300,        # data.ctrl при закрытом схвате
        "gui_invert":       True,
    },

    # ── Датчики AS5600 (для 7 каналов) ──
    "sensor": {
        "port": "COM3",
        "baudrate": 115200,
        "wait_after_connect_ms": 4000,
        "protocol_prefix": "AS5600:",
        "channel_count": 7,

        # Значения по умолчанию (если канал не описан в channels)
        "defaults": {
            "smoothing": 0.30,
            "scale": 1.0,
            "invert": False,
        },

        # Маппинг каналов Arduino → суставы / схват
        #   target: "joint" | "gripper"
        #   joint_index: 0–5 (для target="joint")
        #   enabled: разрешить управление
        #   scale, offset_deg, invert — калибровка
        #   smoothing — коэффициент сглаживания (0.01–0.99)
        "channels": [
            {"channel": 0, "target": "joint", "joint_index": 0,
             "enabled": True, "scale": 1.0, "offset_deg": 0.0,
             "invert": False, "smoothing": 0.30},
            {"channel": 1, "target": "joint", "joint_index": 1,
             "enabled": True, "scale": 1.0, "offset_deg": 0.0,
             "invert": False, "smoothing": 0.30},
            {"channel": 2, "target": "joint", "joint_index": 2,
             "enabled": True, "scale": 1.0, "offset_deg": 0.0,
             "invert": False, "smoothing": 0.30},
            {"channel": 3, "target": "joint", "joint_index": 3,
             "enabled": True, "scale": 1.0, "offset_deg": 0.0,
             "invert": False, "smoothing": 0.30},
            {"channel": 4, "target": "joint", "joint_index": 4,
             "enabled": True, "scale": 1.0, "offset_deg": 0.0,
             "invert": False, "smoothing": 0.30},
            {"channel": 5, "target": "joint", "joint_index": 5,
             "enabled": True, "scale": 1.0, "offset_deg": 0.0,
             "invert": False, "smoothing": 0.30},
            {"channel": 6, "target": "gripper", "joint_index": -1,
             "enabled": True, "scale": 1.0, "offset_deg": 0.0,
             "invert": False, "smoothing": 0.30},
        ],
    },

    # ── Камера ──
    "camera": {
        "default_index": 0,
        "use_virtual": False,
        "frame_delay_ms": 30,
    },

    # ── Детекция ── (без изменений)
    "detection": {
        "min_area": 500,
        "show_contours": True,
        "show_bounding_boxes": True,
        "processing": {
            "blur_kernel": [5, 5],
            "morph_kernel_size": 7,
            "morph_iterations": 2,
        },
        "colors": {
            "Красный": {
                "enabled": True,
                "hsv_ranges": [
                    {"lower": [0, 100, 100],   "upper": [10, 255, 255]},
                    {"lower": [160, 100, 100], "upper": [180, 255, 255]},
                ],
                "draw_bgr": [0, 0, 255],
            },
            "Зелёный": {
                "enabled": True,
                "hsv_ranges": [
                    {"lower": [35, 80, 80], "upper": [85, 255, 255]},
                ],
                "draw_bgr": [0, 255, 0],
            },
            "Синий": {
                "enabled": True,
                "hsv_ranges": [
                    {"lower": [100, 80, 80], "upper": [130, 255, 255]},
                ],
                "draw_bgr": [255, 0, 0],
            },
            "Жёлтый": {
                "enabled": True,
                "hsv_ranges": [
                    {"lower": [20, 100, 100], "upper": [35, 255, 255]},
                ],
                "draw_bgr": [0, 255, 255],
            },
        },
    },

    # ── GUI ── (обновлены метки)
    "gui": {
        "font_monospace": "Consolas",
        "font_size": 13,
        "axis_labels": [
            "J1 Base",   "J2 ShldP", "J3 ShldR",
            "J4 Elbow",  "J5 ForeR", "J6 Wrist",
        ],
        "tcp_labels": ["X", "Y", "Z", "Rx", "Ry", "Rz"],
    },
}

# ═══════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивно объединяет override в base (override приоритетнее)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get_nested(data: dict, dotted_key: str, default=None):
    """Доступ к вложенным ключам через точку: 'a.b.c' → data['a']['b']['c']."""
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def _set_nested(data: dict, dotted_key: str, value):
    """Установка вложенного значения: 'a.b.c' → data['a']['b']['c'] = value."""
    keys = dotted_key.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


# ═══════════════════════════════════════════════════════════════
#  Класс конфигурации (Singleton)
# ═══════════════════════════════════════════════════════════════

class AppConfig:
    """Глобальная конфигурация приложения.

    Загружает config.yaml (или config.json), объединяет со значениями
    по умолчанию.  Предоставляет доступ через точечную нотацию::

        cfg.get("mujoco.rendering.fps")       → 25
        cfg.get("sensor.baudrate")            → 115200
        cfg.get("nonexistent.key", "fallback") → "fallback"
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._data = copy.deepcopy(DEFAULT_CONFIG)
            inst._config_path = None
            inst._load_from_disk()
            cls._instance = inst
        return cls._instance

    # ── Загрузка ──────────────────────────────────────
    def _load_from_disk(self):
        """Ищет config.yaml / config.json и загружает."""
        for name in ("config.yaml", "config.yml", "config.json"):
            if os.path.isfile(name):
                self._config_path = name
                self._load_file(name)
                return
        print("[Config] Файл конфигурации не найден — используются значения по умолчанию.")

    def _load_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    if not YAML_AVAILABLE:
                        print("[Config] ⚠ pyyaml не установлен — pip install pyyaml")
                        return
                    user_data = yaml.safe_load(f) or {}
                else:
                    user_data = json.load(f)
            self._data = _deep_merge(self._data, user_data)
            print(f"[Config] Загружен: {path}")
        except Exception as e:
            print(f"[Config] Ошибка чтения {path}: {e}")

    # ── Публичный API ─────────────────────────────────
    def get(self, key: str, default=None):
        """Получить значение: cfg.get('sensor.baudrate', 9600)"""
        val = _get_nested(self._data, key, default)
        return val if val is not None else default

    def set(self, key: str, value):
        """Установить значение: cfg.set('sensor.port', 'COM5')"""
        _set_nested(self._data, key, value)

    @property
    def data(self) -> dict:
        """Весь конфиг как словарь."""
        return self._data

    def reload(self):
        """Перечитать файл с диска."""
        self._data = copy.deepcopy(DEFAULT_CONFIG)
        if self._config_path:
            self._load_file(self._config_path)
        else:
            self._load_from_disk()

    def save(self, path: str = None):
        """Сохранить текущий конфиг в файл."""
        path = path or self._config_path or "config.yaml"
        try:
            with open(path, "w", encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    if YAML_AVAILABLE:
                        yaml.dump(self._data, f,
                                  allow_unicode=True,
                                  default_flow_style=False,
                                  sort_keys=False)
                    else:
                        # Fallback — JSON
                        path = path.rsplit(".", 1)[0] + ".json"
                        with open(path, "w", encoding="utf-8") as fj:
                            json.dump(self._data, fj, ensure_ascii=False, indent=2)
                else:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
            print(f"[Config] Сохранён: {path}")
            return True
        except Exception as e:
            print(f"[Config] Ошибка сохранения: {e}")
            return False

    def generate_default(self, path: str = "config.yaml"):
        """Создать файл конфигурации с параметрами по умолчанию."""
        backup = self._data
        self._data = copy.deepcopy(DEFAULT_CONFIG)
        result = self.save(path)
        self._data = backup
        return result

    # ── Удобные свойства ──────────────────────────────
    def joint_names(self) -> list:
        """Список имён суставов: ['J1', 'J2', …]"""
        joints = self.get("joints", [])
        return [j.get("name", f"J{i+1}") for i, j in enumerate(joints)]

    def joint_count(self) -> int:
        return len(self.get("joints", []))

    def joint_encoder_cfg(self, index: int) -> dict:
        """Калибровка энкодера для сустава index."""
        joints = self.get("joints", [])
        if 0 <= index < len(joints):
            return joints[index].get("encoder", {"scale": 1.0, "offset_deg": 0.0, "invert": False})
        return {"scale": 1.0, "offset_deg": 0.0, "invert": False}

    def color_configs(self) -> dict:
        """Конфигурации цветов для детекции."""
        return self.get("detection.colors", {})

    def sensor_channel_cfg(self, channel: int) -> dict:
        """Конфигурация одного канала датчика.
        Если канал не описан в channels — генерируется дефолт."""
        channels = self.get("sensor.channels", [])
        for ch in channels:
            if ch.get("channel") == channel:
                return ch
        # Дефолт: канал → сустав напрямую, 7-й → схват
        defaults = self.get("sensor.defaults", {})
        n = self.joint_count()
        return {
            "channel": channel,
            "target": "joint" if channel < n else "gripper",
            "joint_index": channel if channel < n else -1,
            "enabled": True,
            "scale": defaults.get("scale", 1.0),
            "offset_deg": 0.0,
            "invert": defaults.get("invert", False),
            "smoothing": defaults.get("smoothing", 0.30),
        }

    def sensor_channel_count(self) -> int:
        return self.get("sensor.channel_count", 7)

    def __repr__(self):
        return f"<AppConfig path={self._config_path!r} keys={len(self._data)}>"


# ═══════════════════════════════════════════════════════════════
#  Глобальный экземпляр (singleton)
# ═══════════════════════════════════════════════════════════════
cfg = AppConfig()