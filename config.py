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
    # ── Приложение ──
    "application": {
        "window_title": "Управление роботом — Реальный / MuJoCo + AS5600",
        "window_x": 100,
        "window_y": 100,
        "window_width": 1550,
        "window_height": 1050,
        "update_interval_ms": 200,
    },

    # ── Логирование ──
    "logging": {
        "file": "robot_logs.txt",
        "max_entries": 1000,
    },

    # ── Робот ──
    "robot": {
        "default_mode": "simulation",
        "real": {
            "port": "COM39",
            "baudrate": 115200,
        },
        "simulation": {
            "xml_path": "",
        },
        "movement": {
            "default_step": 5,
            "default_mode": "MoveJ",
        },
        "home_position_deg": [0, 0, 0, 0, 0, 0],
    },

    # ── MuJoCo ──
    "mujoco": {
        "rendering": {
            "fps": 25,
            "width": 640,
            "height": 480,
        },
        "cameras": {
            "overview": "overview_cam",
            "hand": "hand_cam",
        },
        "timestep": 0.002,
    },

    # ── Суставы ──
    "joints": [
        {"name": "J1", "mujoco_name": "joint1", "min_deg": -180, "max_deg": 180,
         "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False}},
        {"name": "J2", "mujoco_name": "joint2", "min_deg": -120, "max_deg": 120,
         "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False}},
        {"name": "J3", "mujoco_name": "joint3", "min_deg": -150, "max_deg": 150,
         "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False}},
        {"name": "J4", "mujoco_name": "joint4", "min_deg": -180, "max_deg": 180,
         "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False}},
        {"name": "J5", "mujoco_name": "joint5", "min_deg": -120, "max_deg": 120,
         "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False}},
        {"name": "J6", "mujoco_name": "joint6", "min_deg": -360, "max_deg": 360,
         "encoder": {"scale": 1.0, "offset_deg": 0.0, "invert": False}},
    ],

    # ── Датчик AS5600 ──
    "sensor": {
        "port": "COM3",
        "baudrate": 115200,
        "wait_after_connect_ms": 4000,
        "protocol_prefix": "AS5600:",
        "defaults": {
            "smoothing": 0.30,
            "scale": 1.0,
            "invert": False,
            "joint_index": 0,
        },
    },

    # ── Камера ──
    "camera": {
        "default_index": 0,
        "use_virtual": False,
        "frame_delay_ms": 30,
    },

    # ── Детекция ──
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
                "hsv_ranges": [{"lower": [35, 80, 80], "upper": [85, 255, 255]}],
                "draw_bgr": [0, 255, 0],
            },
            "Синий": {
                "enabled": True,
                "hsv_ranges": [{"lower": [100, 80, 80], "upper": [130, 255, 255]}],
                "draw_bgr": [255, 0, 0],
            },
            "Жёлтый": {
                "enabled": True,
                "hsv_ranges": [{"lower": [20, 100, 100], "upper": [35, 255, 255]}],
                "draw_bgr": [0, 255, 255],
            },
        },
    },

    # ── GUI ──
    "gui": {
        "font_monospace": "Consolas",
        "font_size": 13,
        "axis_labels": ["J1/X", "J2/Y", "J3/Z", "J4/Rx", "J5/Ry", "J6/Rz"],
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

    def __repr__(self):
        return f"<AppConfig path={self._config_path!r} keys={len(self._data)}>"


# ═══════════════════════════════════════════════════════════════
#  Глобальный экземпляр (singleton)
# ═══════════════════════════════════════════════════════════════
cfg = AppConfig()