# robot_control.py

import sys
import math
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QGridLayout, QGroupBox,
    QComboBox, QSpinBox, QFileDialog, QMessageBox, QProgressBar,
    QCheckBox, QSlider, QDoubleSpinBox, QFrame, QTabWidget,
    QScrollArea, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont

from robot_backend import (RobotFactory, RobotMode, RobotState,
                           BaseRobot, MUJOCO_AVAILABLE)
from logger import logger
from config import cfg

# ═══════════════════════════════════════════════════════════════
#  pyserial
# ═══════════════════════════════════════════════════════════════
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  Поток чтения 7× AS5600 через TCA9548A
# ═══════════════════════════════════════════════════════════════
class SensorReaderThread(QThread):
    data_received = pyqtSignal(int, float, int, int, int, str)
    channel_error = pyqtSignal(int, str)
    connection_changed = pyqtSignal(bool, str)

    def __init__(self, port: str, baudrate: int = None, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate or cfg.get("sensor.baudrate", 115200)
        self.running = False
        self._prefix = cfg.get("sensor.protocol_prefix", "AS5600:")
        self._wait_ms = cfg.get("sensor.wait_after_connect_ms", 4000)
        self._serial = None

    def run(self):
        self.running = True
        try:
            self._serial = serial.Serial(
                self.port, self.baudrate, timeout=0.5)
            self.msleep(self._wait_ms)
            self._serial.reset_input_buffer()
            self.connection_changed.emit(True, f"Подключён: {self.port}")
            logger.add(f"[Sensors] Подключён к {self.port}")

            while self.running:
                if self._serial.in_waiting:
                    try:
                        line = (self._serial.readline()
                                .decode('utf-8', errors='ignore').strip())
                        if not line:
                            continue
                        if line.startswith("#"):
                            logger.add(f"[Arduino] {line}")
                            continue
                        self._parse(line)
                    except Exception:
                        pass
                else:
                    self.msleep(2)
        except serial.SerialException as e:
            self.connection_changed.emit(False, f"Ошибка порта: {e}")
            logger.add(f"[Sensors] {e}")
        except Exception as e:
            self.connection_changed.emit(False, f"Ошибка: {e}")
            logger.add(f"[Sensors] {e}")
        finally:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self.connection_changed.emit(False, "Отключён")
            logger.add("[Sensors] Порт закрыт")

    def _parse(self, line: str):
        if not line.startswith(self._prefix):
            return
        payload = line[len(self._prefix):]
        parts = payload.split(",")
        if len(parts) == 6:
            try:
                self.data_received.emit(
                    int(parts[0]), float(parts[1]),
                    int(parts[2]), int(parts[3]),
                    int(parts[4]), parts[5].strip())
            except (ValueError, IndexError):
                pass
        elif len(parts) == 5:
            try:
                self.data_received.emit(
                    0, float(parts[0]), int(parts[1]),
                    int(parts[2]), int(parts[3]), parts[4].strip())
            except (ValueError, IndexError):
                pass
        elif len(parts) == 2:
            try:
                self.channel_error.emit(
                    int(parts[0]), parts[1].strip())
            except ValueError:
                pass

    def send_command(self, cmd: str):
        if self._serial and self._serial.is_open:
            try:
                self._serial.write((cmd.strip() + "\n").encode())
                logger.add(f"[Sensors] → {cmd}")
            except Exception as e:
                logger.add(f"[Sensors] Ошибка отправки: {e}")

    def stop(self):
        self.running = False
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════
#  Поток рендеринга MuJoCo
# ═══════════════════════════════════════════════════════════════
class SimRenderThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, robot: BaseRobot, fps: int = None, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.running = False
        self.fps = fps or cfg.get("mujoco.rendering.fps", 25)

    def run(self):
        self.running = True
        delay = int(1000 / self.fps)
        logger.add("[3D-Вид] Поток запущен")
        while self.running:
            if self.robot and self.robot.is_connected:
                frame = self.robot.get_overview_frame()
                if frame is not None:
                    self.frame_ready.emit(frame)
            self.msleep(delay)
        logger.add("[3D-Вид] Поток остановлен")

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════
#  Поток камеры + детекция
# ═══════════════════════════════════════════════════════════════
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    detection_info = pyqtSignal(list)

    def __init__(self, camera_index=0, robot: BaseRobot = None,
                 parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.robot = robot
        self.running = False
        self.cap = None

        self.min_area = cfg.get("detection.min_area", 500)
        self.show_contours = cfg.get("detection.show_contours", True)
        self.show_bbox = cfg.get("detection.show_bounding_boxes", True)
        self.frame_delay = cfg.get("camera.frame_delay_ms", 30)
        self.use_virtual_camera = cfg.get("camera.use_virtual", False)

        proc = cfg.get("detection.processing", {})
        bk = proc.get("blur_kernel", [5, 5])
        self._blur_kernel = (bk[0], bk[1])
        self._morph_ksize = proc.get("morph_kernel_size", 7)
        self._morph_iters = proc.get("morph_iterations", 2)

        self.color_ranges = {}
        self.draw_colors = {}
        self.enabled_colors = {}
        for name, c_cfg in cfg.color_configs().items():
            ranges = []
            for r in c_cfg.get("hsv_ranges", []):
                ranges.append(
                    (np.array(r["lower"]), np.array(r["upper"])))
            self.color_ranges[name] = ranges
            self.draw_colors[name] = tuple(
                c_cfg.get("draw_bgr", [255, 255, 255]))
            self.enabled_colors[name] = c_cfg.get("enabled", True)

    def set_min_area(self, area: int):
        self.min_area = area

    def set_color_enabled(self, color_name: str, enabled: bool):
        if color_name in self.enabled_colors:
            self.enabled_colors[color_name] = enabled

    def run(self):
        self.running = True
        if not self.use_virtual_camera:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.add(
                    f"[Камера] Не удалось открыть #{self.camera_index}")
                self.running = False
                return
            logger.add(f"[Камера] Подключена #{self.camera_index}")
        else:
            logger.add("[Камера] Виртуальная (MuJoCo)")

        while self.running:
            frame = None
            if self.use_virtual_camera and self.robot:
                frame = self.robot.get_camera_frame()
            elif self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    frame = None
            if frame is not None:
                processed, detections = self._process_frame(frame)
                self.frame_ready.emit(processed)
                if detections:
                    self.detection_info.emit(detections)
            self.msleep(self.frame_delay)
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def _process_frame(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, self._blur_kernel, 0)
        detections = []
        overlay = frame.copy()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._morph_ksize, self._morph_ksize))
        for color_name, ranges in self.color_ranges.items():
            if not self.enabled_colors.get(color_name, False):
                continue
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel,
                iterations=self._morph_iters)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel,
                iterations=self._morph_iters)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            draw_color = self.draw_colors[color_name]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detections.append({
                    "color": color_name, "area": int(area),
                    "cx": cx, "cy": cy})
                if self.show_contours:
                    cv2.drawContours(
                        overlay, [cnt], -1, draw_color, 2)
                if self.show_bbox:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(
                        overlay, (x, y), (x + w, y + h),
                        draw_color, 2)
                cv2.putText(
                    overlay, f"{color_name}: {area:.0f}",
                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, draw_color, 2)
        cv2.putText(
            overlay, f"Objects: {len(detections)}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return overlay, detections


# ═══════════════════════════════════════════════════════════════
#  Главное окно
# ═══════════════════════════════════════════════════════════════
class RobotControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robot: BaseRobot = None
        self.camera_thread = None
        self.sim_render_thread = None

        # AS5600 × 7
        self.sensor_thread: SensorReaderThread = None
        self.sensor_connected = False
        self._ch_data = {}
        self._ch_widgets = {}

        self.setWindowTitle(cfg.get(
            "application.window_title", "Управление роботом"))
        self.setGeometry(
            cfg.get("application.window_x", 100),
            cfg.get("application.window_y", 100),
            cfg.get("application.window_width", 1550),
            cfg.get("application.window_height", 1050))
        self.init_ui()
        self.setup_timer()

    # ══════════════════════════════════════════════════════════
    #  ПОСТРОЕНИЕ ИНТЕРФЕЙСА
    # ══════════════════════════════════════════════════════════
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        mono = cfg.get("gui.font_monospace", "Consolas")
        fsz = cfg.get("gui.font_size", 13)
        joint_names = cfg.joint_names()
        axis_labels = cfg.get("gui.axis_labels",
                              ["J1/X", "J2/Y", "J3/Z",
                               "J4/Rx", "J5/Ry", "J6/Rz"])
        tcp_labels = cfg.get("gui.tcp_labels",
                             ["X", "Y", "Z", "Rx", "Ry", "Rz"])

        # ════════════ ЛЕВАЯ ПАНЕЛЬ ════════════
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 2)

        # ── 1. Режим работы (всегда видим) ────
        left_panel.addWidget(self._build_mode_group(mono, fsz))

        # ── 2. Кнопки управления (всегда видны) ──
        left_panel.addWidget(self._build_ctrl_group())

        # ── 3. ВКЛАДКИ: Управление / Датчики ──
        self.left_tabs = QTabWidget()
        self.left_tabs.setStyleSheet(f"font-size: {fsz}px;")

        # --- Вкладка «Управление» ---
        ctrl_tab = QWidget()
        ctrl_tab_layout = QVBoxLayout(ctrl_tab)
        ctrl_tab_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_tab_layout.addWidget(
            self._build_joy_group(axis_labels, mono))
        ctrl_tab_layout.addWidget(self._build_grip_group())
        ctrl_tab_layout.addStretch()
        self.left_tabs.addTab(ctrl_tab, "🎮 Управление")

        # --- Вкладка «Датчики» ---
        sensor_tab = QWidget()
        sensor_tab_layout = QVBoxLayout(sensor_tab)
        sensor_tab_layout.setContentsMargins(4, 4, 4, 4)
        sensor_tab_layout.addWidget(
            self._build_sensor_group(mono, joint_names))
        sensor_tab_layout.addStretch()
        self.left_tabs.addTab(sensor_tab, "🔄 Датчики AS5600")

        left_panel.addWidget(self.left_tabs)

        # ── 4. Статус (всегда видим) ──────────
        self.status_label = QLabel("НЕ ПОДКЛЮЧЕНО")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: gray; color: white; "
            "font-size: 22px; font-weight: bold; padding: 15px;")
        left_panel.addWidget(self.status_label)

        # ════════════ ПРАВАЯ ПАНЕЛЬ ════════════
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 3)

        right_panel.addWidget(
            self._build_pose_group(tcp_labels, mono, fsz))
        right_panel.addWidget(
            self._build_torque_group(joint_names, mono))
        right_panel.addWidget(
            self._build_view_tabs(mono, fsz))
        right_panel.addWidget(self._build_log_group())

    # ──────────────────────────────────────────────────
    #  Билдеры секций
    # ──────────────────────────────────────────────────

    def _build_mode_group(self, mono, fsz) -> QGroupBox:
        grp = QGroupBox("⚙ Режим работы")
        grp.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; }")
        lay = QVBoxLayout()

        # Цель подключения (создаём первым!)
        conn_row = QHBoxLayout()
        self.conn_target_edit = QLineEdit(
            cfg.get("robot.real.port", "COM39"))
        self.conn_target_edit.setPlaceholderText(
            "COM-порт или путь к XML")
        conn_row.addWidget(QLabel("Цель:"))
        conn_row.addWidget(self.conn_target_edit)

        # Режим
        mode_row = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("🤖 Реальный робот (MCX)")
        if MUJOCO_AVAILABLE:
            self.mode_selector.addItem("🖥 Симуляция (MuJoCo)")
        else:
            self.mode_selector.addItem("🖥 MuJoCo ⚠ НЕДОСТУПНА")
            self.mode_selector.model().item(1).setEnabled(False)
        self.mode_selector.setStyleSheet(
            f"font-size: {fsz}px; padding: 5px;")
        mode_row.addWidget(QLabel("Режим:"))
        mode_row.addWidget(self.mode_selector)

        self.mode_selector.currentIndexChanged.connect(
            self._on_mode_selector_changed)

        default_mode = cfg.get("robot.default_mode", "simulation")
        if default_mode == "simulation" and MUJOCO_AVAILABLE:
            self.mode_selector.setCurrentIndex(1)

        lay.addLayout(mode_row)
        lay.addLayout(conn_row)

        # Кнопки подключения
        btns = QHBoxLayout()
        self.btn_connect = QPushButton("🔌 Подключить")
        self.btn_connect.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px;")
        self.btn_connect.clicked.connect(self.connect_robot)
        self.btn_disconnect = QPushButton("🔌 Отключить")
        self.btn_disconnect.setStyleSheet(
            "background-color: #757575; color: white; "
            "font-weight: bold; padding: 10px;")
        self.btn_disconnect.clicked.connect(self.disconnect_robot)
        self.btn_disconnect.setEnabled(False)
        btns.addWidget(self.btn_connect)
        btns.addWidget(self.btn_disconnect)
        lay.addLayout(btns)

        self.mode_indicator = QLabel("Не подключено")
        self.mode_indicator.setAlignment(Qt.AlignCenter)
        self.mode_indicator.setStyleSheet(
            "background-color: #666; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 4px;")
        lay.addWidget(self.mode_indicator)
        grp.setLayout(lay)
        return grp

    def _build_ctrl_group(self) -> QGroupBox:
        grp = QGroupBox("Управление")
        lay = QGridLayout()
        self.btn_on = QPushButton("ВКЛ")
        self.btn_off = QPushButton("ВЫКЛ")
        self.btn_pause = QPushButton("ПАУЗА")
        self.btn_resume = QPushButton("ПРОДОЛЖИТЬ")
        self.btn_emergency = QPushButton("⛔ СТОП")
        self.btn_home = QPushButton("🏠 HOME")
        self.btn_on.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px;")
        self.btn_emergency.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; padding: 8px;")
        self.btn_home.setStyleSheet(
            "background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 8px;")
        for b in [self.btn_off, self.btn_pause, self.btn_resume]:
            b.setStyleSheet("font-weight: bold; padding: 8px;")
        lay.addWidget(self.btn_on, 0, 0)
        lay.addWidget(self.btn_off, 0, 1)
        lay.addWidget(self.btn_pause, 1, 0)
        lay.addWidget(self.btn_resume, 1, 1)
        lay.addWidget(self.btn_home, 2, 0)
        lay.addWidget(self.btn_emergency, 2, 1)
        grp.setLayout(lay)

        self.btn_on.clicked.connect(
            lambda: logger.add("Робот включён"))
        self.btn_off.clicked.connect(self._on_btn_off)
        self.btn_pause.clicked.connect(
            lambda: self._safe_call(lambda: self.robot.pause()))
        self.btn_resume.clicked.connect(
            lambda: self._safe_call(lambda: self.robot.resume()))
        self.btn_emergency.clicked.connect(
            lambda: self._safe_call(
                lambda: self.robot.emergency_stop()))
        self.btn_home.clicked.connect(
            lambda: self._safe_call(
                lambda: self.robot.move_to_home()))
        return grp

    def _build_joy_group(self, axis_labels, mono) -> QGroupBox:
        grp = QGroupBox("Ручное управление")
        lay = QGridLayout()

        self.move_mode_combo = QComboBox()
        self.move_mode_combo.addItems(
            ["MoveJ (по суставам)", "MoveL (линейно)"])
        dm = cfg.get("robot.movement.default_mode", "MoveJ")
        if dm == "MoveL":
            self.move_mode_combo.setCurrentIndex(1)
        lay.addWidget(QLabel("Режим:"), 0, 0)
        lay.addWidget(self.move_mode_combo, 0, 1, 1, 3)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 50)
        self.step_spin.setValue(
            cfg.get("robot.movement.default_step", 5))
        self.step_spin.setSuffix("°") 
        lay.addWidget(QLabel("Шаг:"), 1, 0)
        lay.addWidget(self.step_spin, 1, 1, 1, 3)

        self.joy_val_labels = []
        for i, name in enumerate(axis_labels):
            btn_m = QPushButton("−")
            btn_m.setFixedWidth(40)
            val_lbl = QLabel("0.00")
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet(
                "background-color: #eee; border: 1px solid #ccc; "
                "font-weight: bold;")
            val_lbl.setFixedWidth(70)
            self.joy_val_labels.append(val_lbl)
            btn_p = QPushButton("+")
            btn_p.setFixedWidth(40)
            btn_m.clicked.connect(
                lambda _, idx=i: self.move_axis(idx, -1))
            btn_p.clicked.connect(
                lambda _, idx=i: self.move_axis(idx, 1))
            lay.addWidget(QLabel(name), i + 2, 0)
            lay.addWidget(btn_m, i + 2, 1)
            lay.addWidget(val_lbl, i + 2, 2)
            lay.addWidget(btn_p, i + 2, 3)
        grp.setLayout(lay)
        return grp

    def _build_grip_group(self) -> QGroupBox:
        grp = QGroupBox("Схват")
        lay = QHBoxLayout()
        btn_open = QPushButton("Открыть")
        btn_close = QPushButton("Закрыть")
        btn_open.clicked.connect(
            lambda: self._gripper_command("open"))
        btn_close.clicked.connect(
            lambda: self._gripper_command("close"))

        self.gripper_slider = QSlider(Qt.Horizontal)
        self.gripper_slider.setRange(0, 100)
        self.gripper_slider.setValue(0)
        self.gripper_slider.setToolTip(
            "0 = открыт, 100 = закрыт")
        self.gripper_slider.valueChanged.connect(
            self._on_gripper_slider)

        self.gripper_label = QLabel("0%")
        self.gripper_label.setFixedWidth(35)
        self.gripper_label.setAlignment(Qt.AlignCenter)

        lay.addWidget(btn_open)
        lay.addWidget(self.gripper_slider)
        lay.addWidget(self.gripper_label)
        lay.addWidget(btn_close)
        grp.setLayout(lay)
        return grp

    def _build_sensor_group(self, mono, joint_names) -> QGroupBox:
        grp = QGroupBox("🔄 AS5600 × 7 — Датчики суставов")
        grp.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 13px; }")
        lay = QVBoxLayout()

        # --- Подключение ---
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Порт:"))
        self.sensor_port_combo = QComboBox()
        self.sensor_port_combo.setEditable(True)
        self.sensor_port_combo.setMinimumWidth(100)
        conn_row.addWidget(self.sensor_port_combo)

        self.btn_refresh_ports = QPushButton("🔄")
        self.btn_refresh_ports.setFixedWidth(28)
        self.btn_refresh_ports.setToolTip(
            "Обновить список COM-портов")
        self.btn_refresh_ports.clicked.connect(
            self._refresh_sensor_ports)
        conn_row.addWidget(self.btn_refresh_ports)

        self.btn_sensor_connect = QPushButton("▶")
        self.btn_sensor_connect.setFixedWidth(32)
        self.btn_sensor_connect.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold;")
        self.btn_sensor_connect.setToolTip("Подключить")
        self.btn_sensor_connect.clicked.connect(
            self.connect_sensor)
        conn_row.addWidget(self.btn_sensor_connect)

        self.btn_sensor_disconnect = QPushButton("■")
        self.btn_sensor_disconnect.setFixedWidth(32)
        self.btn_sensor_disconnect.setToolTip("Отключить")
        self.btn_sensor_disconnect.clicked.connect(
            self.disconnect_sensor)
        self.btn_sensor_disconnect.setEnabled(False)
        conn_row.addWidget(self.btn_sensor_disconnect)

        self.btn_sensor_scan = QPushButton("SCAN")
        self.btn_sensor_scan.setFixedWidth(50)
        self.btn_sensor_scan.setToolTip(
            "Пересканировать датчики")
        self.btn_sensor_scan.clicked.connect(self._sensor_scan)
        self.btn_sensor_scan.setEnabled(False)
        conn_row.addWidget(self.btn_sensor_scan)

        conn_row.addStretch()
        lay.addLayout(conn_row)

        # --- Таблица каналов ---
        ch_count = cfg.sensor_channel_count()
        self._ch_data = {}
        self._ch_widgets = {}

        ch_grid = QGridLayout()
        ch_grid.setSpacing(3)

        headers = ["", "Канал → Цель", "Δ Угол",
                    "", "RAW", "AGC", "Магнит", ""]
        for col, hdr in enumerate(headers):
            lbl = QLabel(hdr)
            lbl.setStyleSheet("font-size: 10px; color: #888;")
            ch_grid.addWidget(lbl, 0, col)

        for i in range(ch_count):
            row = i + 1
            ch_cfg = cfg.sensor_channel_cfg(i)
            target_label = self._channel_target_label(ch_cfg)

            self._ch_data[i] = {
                "deg": 0.0, "smoothed": 0.0,
                "zero": ch_cfg.get("offset_deg", 0.0),
                "raw": 0, "agc": 0, "mag": 0,
                "status": "—", "online": False,
            }

            # Чекбокс
            chk = QCheckBox()
            chk.setChecked(ch_cfg.get("enabled", True))
            chk.setToolTip(f"Включить CH{i}")
            ch_grid.addWidget(chk, row, 0)

            # CH → Цель
            ch_lbl = QLabel(f"CH{i} → {target_label}")
            ch_lbl.setStyleSheet(
                "font-size: 11px; font-weight: bold;")
            ch_lbl.setFixedWidth(95)
            ch_grid.addWidget(ch_lbl, row, 1)

            # Δ Угол
            angle_lbl = QLabel("  —  ")
            angle_lbl.setAlignment(
                Qt.AlignRight | Qt.AlignVCenter)
            angle_lbl.setStyleSheet(
                f"font-family: {mono}; font-size: 12px; "
                f"background-color: #1a1a2e; color: #555; "
                f"padding: 2px 4px;")
            angle_lbl.setFixedWidth(70)
            ch_grid.addWidget(angle_lbl, row, 2)

            # Статус-индикатор
            status_lbl = QLabel("●")
            status_lbl.setFixedWidth(16)
            status_lbl.setAlignment(Qt.AlignCenter)
            status_lbl.setStyleSheet(
                "color: #555; font-size: 12px;")
            status_lbl.setToolTip("нет данных")
            ch_grid.addWidget(status_lbl, row, 3)

            # RAW
            raw_lbl = QLabel("—")
            raw_lbl.setFixedWidth(42)
            raw_lbl.setAlignment(Qt.AlignCenter)
            raw_lbl.setStyleSheet(
                f"font-family: {mono}; font-size: 10px; "
                f"color: #777;")
            ch_grid.addWidget(raw_lbl, row, 4)

            # AGC
            agc_lbl = QLabel("—")
            agc_lbl.setFixedWidth(30)
            agc_lbl.setAlignment(Qt.AlignCenter)
            agc_lbl.setStyleSheet(
                f"font-family: {mono}; font-size: 10px; "
                f"color: #777;")
            ch_grid.addWidget(agc_lbl, row, 5)

            # Магнит
            mag_lbl = QLabel("—")
            mag_lbl.setFixedWidth(35)
            mag_lbl.setAlignment(Qt.AlignCenter)
            mag_lbl.setStyleSheet(
                f"font-family: {mono}; font-size: 10px; "
                f"color: #777;")
            ch_grid.addWidget(mag_lbl, row, 6)

            # Кнопка нуля
            zero_btn = QPushButton("⓪")
            zero_btn.setFixedSize(24, 22)
            zero_btn.setToolTip(f"Установить ноль CH{i}")
            zero_btn.clicked.connect(
                lambda _, c=i: self._set_channel_zero(c))
            ch_grid.addWidget(zero_btn, row, 7)

            self._ch_widgets[i] = {
                "check": chk, "angle": angle_lbl,
                "status": status_lbl, "zero_btn": zero_btn,
                "label": ch_lbl, "raw": raw_lbl,
                "agc": agc_lbl, "mag": mag_lbl,
            }

        lay.addLayout(ch_grid)

        # --- Глобальные контролы ---
        global_row = QHBoxLayout()
        self.chk_sensor_enable = QCheckBox("Управление ВКЛ")
        self.chk_sensor_enable.setStyleSheet(
            "font-weight: bold; color: #2196F3;")
        global_row.addWidget(self.chk_sensor_enable)

        btn_all_zero = QPushButton("⓪ Все нули")
        btn_all_zero.setFixedWidth(80)
        btn_all_zero.clicked.connect(self._set_all_zeros)
        global_row.addWidget(btn_all_zero)
        global_row.addStretch()
        lay.addLayout(global_row)

        # Сглаживание
        sp = int(cfg.get("sensor.defaults.smoothing", 0.30) * 100)
        sm_row = QHBoxLayout()
        sm_row.addWidget(QLabel("Сглаж:"))
        self.sensor_smooth_slider = QSlider(Qt.Horizontal)
        self.sensor_smooth_slider.setRange(1, 99)
        self.sensor_smooth_slider.setValue(sp)
        sm_row.addWidget(self.sensor_smooth_slider)
        self.sensor_smooth_val = QLabel(f"{sp / 100:.2f}")
        self.sensor_smooth_val.setFixedWidth(35)
        self.sensor_smooth_slider.valueChanged.connect(
            lambda v: self.sensor_smooth_val.setText(
                f"{v / 100:.2f}"))
        sm_row.addWidget(self.sensor_smooth_val)
        lay.addLayout(sm_row)

        # Статус
        self.sensor_status_label = QLabel("Не подключено")
        self.sensor_status_label.setStyleSheet(
            f"font-family: {mono}; font-size: 10px; color: #888;")
        lay.addWidget(self.sensor_status_label)

        if not SERIAL_AVAILABLE:
            w = QLabel("⚠ pip install pyserial")
            w.setStyleSheet(
                "color: #f44336; font-weight: bold;")
            lay.addWidget(w)
            self.btn_sensor_connect.setEnabled(False)

        grp.setLayout(lay)
        self._refresh_sensor_ports()
        return grp

    def _build_pose_group(self, tcp_labels, mono, fsz) -> QGroupBox:
        grp = QGroupBox("Текущая поза TCP")
        lay = QGridLayout()
        self.pos_labels = []
        for i, name in enumerate(tcp_labels):
            lay.addWidget(QLabel(f"{name}:"),
                          i // 3, (i % 3) * 2)
            lbl = QLabel("0.000")
            lbl.setStyleSheet(
                f"font-family: {mono}; font-size: {fsz}px;")
            lay.addWidget(lbl, i // 3, (i % 3) * 2 + 1)
            self.pos_labels.append(lbl)
        grp.setLayout(lay)
        return grp

    def _build_torque_group(self, joint_names, mono) -> QGroupBox:
        self.torque_group = QGroupBox("Моменты (Н·м)")
        lay = QGridLayout()
        self.torque_labels = []
        for i in range(cfg.joint_count()):
            lay.addWidget(QLabel(f"{joint_names[i]}:"),
                          i // 3, (i % 3) * 2)
            lbl = QLabel("0.000")
            lbl.setStyleSheet(f"font-family: {mono};")
            lay.addWidget(lbl, i // 3, (i % 3) * 2 + 1)
            self.torque_labels.append(lbl)
        self.torque_group.setLayout(lay)
        self.torque_group.setVisible(False)
        return self.torque_group

    def _build_view_tabs(self, mono, fsz) -> QTabWidget:
        self.view_tabs = QTabWidget()
        self.view_tabs.setStyleSheet(f"font-size: {fsz}px;")

        # --- 3D ---
        sim_tab = QWidget()
        sim_lay = QVBoxLayout(sim_tab)
        self.sim_view_label = QLabel(
            "Подключите MuJoCo для 3D-визуализации")
        self.sim_view_label.setMinimumSize(
            cfg.get("mujoco.rendering.width", 640),
            cfg.get("mujoco.rendering.height", 480))
        self.sim_view_label.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; "
            "font-size: 16px;")
        self.sim_view_label.setAlignment(Qt.AlignCenter)
        sim_lay.addWidget(self.sim_view_label)
        self.sim_info_label = QLabel("")
        self.sim_info_label.setStyleSheet(
            f"font-family: {mono}; font-size: 12px; padding: 4px;")
        sim_lay.addWidget(self.sim_info_label)
        self.view_tabs.addTab(sim_tab, "🖥 3D Симуляция")

        # --- Камера ---
        cam_tab = QWidget()
        cam_lay = QVBoxLayout(cam_tab)

        cam_set = QHBoxLayout()
        cam_set.addWidget(QLabel("Камера:"))
        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 10)
        self.cam_index_spin.setValue(
            cfg.get("camera.default_index", 0))
        cam_set.addWidget(self.cam_index_spin)
        self.chk_virtual_cam = QCheckBox("Виртуальная (MuJoCo)")
        self.chk_virtual_cam.setChecked(
            cfg.get("camera.use_virtual", False))
        cam_set.addWidget(self.chk_virtual_cam)
        self.btn_cam_start = QPushButton("▶ Старт")
        self.btn_cam_start.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold;")
        self.btn_cam_start.clicked.connect(self.start_camera)
        cam_set.addWidget(self.btn_cam_start)
        self.btn_cam_stop = QPushButton("■ Стоп")
        self.btn_cam_stop.clicked.connect(self.stop_camera)
        self.btn_cam_stop.setEnabled(False)
        cam_set.addWidget(self.btn_cam_stop)
        cam_set.addStretch()
        cam_lay.addLayout(cam_set)

        det_set = QHBoxLayout()
        det_set.addWidget(QLabel("Мин.площадь:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(50, 50000)
        self.min_area_spin.setValue(
            cfg.get("detection.min_area", 500))
        self.min_area_spin.setSingleStep(100)
        self.min_area_spin.valueChanged.connect(
            self._on_min_area_changed)
        det_set.addWidget(self.min_area_spin)

        self.color_checks = {}
        for cn, cd in cfg.color_configs().items():
            bgr = cd.get("draw_bgr", [255, 255, 255])
            cb = QCheckBox(cn)
            cb.setChecked(cd.get("enabled", True))
            r, g, b = bgr[2], bgr[1], bgr[0]
            cb.setStyleSheet(
                f"color: rgb({r},{g},{b}); font-weight: bold;")
            cb.stateChanged.connect(
                lambda s, n=cn: self._on_color_toggle(n, s))
            det_set.addWidget(cb)
            self.color_checks[cn] = cb
        det_set.addStretch()
        cam_lay.addLayout(det_set)

        self.video_label = QLabel("Камера не подключена")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; "
            "font-size: 16px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        cam_lay.addWidget(self.video_label)

        self.detection_label = QLabel("Объекты: —")
        self.detection_label.setStyleSheet(
            f"font-family: {mono}; font-size: 12px;")
        self.detection_label.setWordWrap(True)
        cam_lay.addWidget(self.detection_label)
        self.view_tabs.addTab(cam_tab, "📷 Камера + Детекция")
        return self.view_tabs

    def _build_log_group(self) -> QGroupBox:
        grp = QGroupBox("Логи")
        lay = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        lay.addWidget(self.log_view)
        save_row = QHBoxLayout()
        self.path_edit = QLineEdit(
            cfg.get("logging.file", "robot_logs.txt"))
        btn_save = QPushButton("Сохранить")
        btn_save.clicked.connect(self.save_logs)
        save_row.addWidget(self.path_edit)
        save_row.addWidget(btn_save)
        lay.addLayout(save_row)
        grp.setLayout(lay)
        return grp

    # ══════════════════════════════════════════════════════════
    #  Подключение / отключение РОБОТА
    # ══════════════════════════════════════════════════════════
    def connect_robot(self):
        if self.robot and self.robot.is_connected:
            self.disconnect_robot()
        idx = self.mode_selector.currentIndex()
        mode = RobotMode.REAL if idx == 0 else RobotMode.SIMULATION
        target = self.conn_target_edit.text().strip()
        try:
            self.robot = RobotFactory.create(mode)
            success = self.robot.connect(
                target, enable_rendering=True)
            if success:
                self.btn_connect.setEnabled(False)
                self.btn_disconnect.setEnabled(True)
                self.mode_selector.setEnabled(False)
                mt = ("🤖 Реальный" if mode == RobotMode.REAL
                      else "🖥 MuJoCo")
                self.mode_indicator.setText(f"✅ {mt}")
                self.mode_indicator.setStyleSheet(
                    "background-color: #4CAF50; color: white; "
                    "font-size: 13px; padding: 8px; "
                    "border-radius: 4px;")
                self.torque_group.setVisible(
                    mode == RobotMode.SIMULATION)
                self.chk_virtual_cam.setChecked(
                    mode == RobotMode.SIMULATION)
                if mode == RobotMode.SIMULATION:
                    self._start_sim_render()
                    self.view_tabs.setCurrentIndex(0)
                logger.add(f"Подключено: {mt}")
            else:
                QMessageBox.warning(
                    self, "Ошибка", "Не удалось подключиться")
                self.robot = None
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка:\n{e}")
            logger.add(f"Ошибка: {e}")

    def disconnect_robot(self):
        self.chk_sensor_enable.setChecked(False)
        self._stop_sim_render()
        self.stop_camera()
        if self.robot:
            self.robot.disconnect()
            self.robot = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.mode_selector.setEnabled(True)
        self.mode_indicator.setText("Не подключено")
        self.mode_indicator.setStyleSheet(
            "background-color: #666; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 4px;")
        self.status_label.setText("НЕ ПОДКЛЮЧЕНО")
        self.status_label.setStyleSheet(
            "background-color: gray; color: white; "
            "font-size: 22px; padding: 15px;")
        self.sim_view_label.setText(
            "Подключите MuJoCo для 3D-визуализации")
        self.sim_view_label.setPixmap(QPixmap())
        self.sim_info_label.setText("")

    def _on_mode_selector_changed(self, index):
        if index == 0:
            self.conn_target_edit.setText(
                cfg.get("robot.real.port", "COM39"))
            self.conn_target_edit.setPlaceholderText("COM-порт")
        else:
            self.conn_target_edit.setText(
                cfg.get("robot.simulation.xml_path", ""))
            self.conn_target_edit.setPlaceholderText(
                "Путь к XML (пусто = встроенная)")

    # ══════════════════════════════════════════════════════════
    #  3D-визуализация
    # ══════════════════════════════════════════════════════════
    def _start_sim_render(self):
        if self.sim_render_thread:
            self._stop_sim_render()
        self.sim_render_thread = SimRenderThread(
            robot=self.robot, parent=self)
        self.sim_render_thread.frame_ready.connect(
            self._display_sim_frame)
        self.sim_render_thread.start()

    def _stop_sim_render(self):
        if self.sim_render_thread:
            self.sim_render_thread.stop()
            self.sim_render_thread = None

    def _display_sim_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w,
                       QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.sim_view_label.width(),
            self.sim_view_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.sim_view_label.setPixmap(pix)
        if self.robot and self.robot.is_connected:
            info = self.robot.get_info()
            self.sim_info_label.setText(
                f"Время: {info.get('sim_time', 0):.2f}с  |  "
                f"Состояние: {info.get('state', '?')}")

    # ══════════════════════════════════════════════════════════
    #  Камера
    # ══════════════════════════════════════════════════════════
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        idx = self.cam_index_spin.value()
        virt = self.chk_virtual_cam.isChecked()
        self.camera_thread = CameraThread(
            camera_index=idx, robot=self.robot, parent=self)
        self.camera_thread.use_virtual_camera = virt
        self.camera_thread.min_area = self.min_area_spin.value()
        for cn, cb in self.color_checks.items():
            self.camera_thread.set_color_enabled(
                cn, cb.isChecked())
        self.camera_thread.frame_ready.connect(
            self._display_cam_frame)
        self.camera_thread.detection_info.connect(
            self._update_detection_info)
        self.camera_thread.start()
        self.btn_cam_start.setEnabled(False)
        self.btn_cam_stop.setEnabled(True)

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.btn_cam_start.setEnabled(True)
        self.btn_cam_stop.setEnabled(False)
        self.video_label.setText("Камера отключена")
        self.video_label.setPixmap(QPixmap())

    def _display_cam_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w,
                       QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _update_detection_info(self, detections: list):
        if not detections:
            self.detection_label.setText("Объекты: —")
            return
        by_c = {}
        for d in detections:
            by_c.setdefault(d["color"], []).append(d)
        parts = [f"{c}: {len(v)} шт."
                 for c, v in by_c.items()]
        self.detection_label.setText(
            f"Объектов: {len(detections)}  |  "
            + "  ".join(parts))

    def _on_min_area_changed(self, value):
        if self.camera_thread:
            self.camera_thread.set_min_area(value)

    def _on_color_toggle(self, color_name, state):
        if self.camera_thread:
            self.camera_thread.set_color_enabled(
                color_name, state == Qt.Checked)

    # ══════════════════════════════════════════════════════════
    #  Схват
    # ══════════════════════════════════════════════════════════
    def _gripper_command(self, action: str):
        if not self.robot or not self.robot.is_connected:
            logger.add("Робот не подключён!")
            return
        inv = cfg.get("gripper.gui_invert", False)
        if action == "open":
            val = 1.0 if inv else 0.0
        else:
            val = 0.0 if inv else 1.0
        try:
            self.robot.set_gripper(val)
            sp = (int((1.0 - val) * 100) if inv
                  else int(val * 100))
            self.gripper_slider.blockSignals(True)
            self.gripper_slider.setValue(sp)
            self.gripper_slider.blockSignals(False)
            self.gripper_label.setText(f"{sp}%")
        except Exception as e:
            logger.add(f"[Схват] Ошибка: {e}")

    def _on_gripper_slider(self, sv: int):
        if not self.robot or not self.robot.is_connected:
            return
        inv = cfg.get("gripper.gui_invert", False)
        f = sv / 100.0
        val = (1.0 - f) if inv else f
        try:
            self.robot.set_gripper(val)
            self.gripper_label.setText(f"{sv}%")
        except Exception as e:
            logger.add(f"[Схват] Ошибка: {e}")

    # ══════════════════════════════════════════════════════════
    #  AS5600 × 7
    # ══════════════════════════════════════════════════════════
    def _channel_target_label(self, ch_cfg: dict) -> str:
        target = ch_cfg.get("target", "joint")
        if target == "gripper":
            return "Grip"
        ji = ch_cfg.get("joint_index",
                        ch_cfg.get("channel", 0))
        names = cfg.joint_names()
        if 0 <= ji < len(names):
            return names[ji]
        return f"J{ji + 1}"

    def _refresh_sensor_ports(self):
        if not SERIAL_AVAILABLE:
            return
        cur = self.sensor_port_combo.currentText()
        self.sensor_port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in sorted(ports, key=lambda x: x.device):
            self.sensor_port_combo.addItem(
                p.device, userData=p.device)
        if self.sensor_port_combo.count() == 0:
            self.sensor_port_combo.addItem(
                cfg.get("sensor.port", "COM3"))
        idx = self.sensor_port_combo.findText(cur)
        if idx >= 0:
            self.sensor_port_combo.setCurrentIndex(idx)

    def connect_sensor(self):
        if not SERIAL_AVAILABLE:
            QMessageBox.warning(
                self, "Ошибка", "pip install pyserial")
            return
        if self.sensor_thread is not None:
            self.disconnect_sensor()
        port = self.sensor_port_combo.currentData()
        if port is None:
            port = (self.sensor_port_combo.currentText()
                    .split(" ")[0].strip())
        rp = self.conn_target_edit.text().strip().upper()
        if (port.upper() == rp
                and self.robot and self.robot.is_connected):
            QMessageBox.warning(
                self, "Конфликт", f"Порт {port} занят роботом!")
            return
        self.sensor_thread = SensorReaderThread(
            port, parent=self)
        self.sensor_thread.data_received.connect(
            self._on_sensor_data)
        self.sensor_thread.channel_error.connect(
            self._on_sensor_error)
        self.sensor_thread.connection_changed.connect(
            self._on_sensor_connection)
        self.sensor_thread.start()
        self.btn_sensor_connect.setEnabled(False)
        self.btn_sensor_disconnect.setEnabled(True)
        self.btn_sensor_scan.setEnabled(True)
        self.sensor_port_combo.setEnabled(False)
        self.btn_refresh_ports.setEnabled(False)

    def disconnect_sensor(self):
        self.chk_sensor_enable.setChecked(False)
        if self.sensor_thread is not None:
            self.sensor_thread.stop()
            self.sensor_thread = None
        self.sensor_connected = False
        self.btn_sensor_connect.setEnabled(True)
        self.btn_sensor_disconnect.setEnabled(False)
        self.btn_sensor_scan.setEnabled(False)
        self.sensor_port_combo.setEnabled(True)
        self.btn_refresh_ports.setEnabled(True)
        mono = cfg.get("gui.font_monospace", "Consolas")
        for ch, w in self._ch_widgets.items():
            w["angle"].setText("  —  ")
            w["angle"].setStyleSheet(
                f"font-family: {mono}; font-size: 12px; "
                f"background-color: #1a1a2e; color: #555; "
                f"padding: 2px 4px;")
            w["status"].setStyleSheet(
                "color: #555; font-size: 12px;")
            w["status"].setToolTip("нет данных")
            w["raw"].setText("—")
            w["agc"].setText("—")
            w["mag"].setText("—")
            self._ch_data[ch]["online"] = False
        self.sensor_status_label.setText("Не подключено")

    def _on_sensor_connection(self, connected: bool, msg: str):
        self.sensor_connected = connected
        self.sensor_status_label.setText(msg)
        if not connected and self.sensor_thread is not None:
            self.btn_sensor_connect.setEnabled(True)
            self.btn_sensor_disconnect.setEnabled(False)
            self.btn_sensor_scan.setEnabled(False)
            self.sensor_port_combo.setEnabled(True)
            self.btn_refresh_ports.setEnabled(True)
            self.sensor_thread = None

    def _on_sensor_data(self, ch: int, deg: float,
                        raw: int, agc: int,
                        mag: int, status: str):
        if ch not in self._ch_data:
            return
        d = self._ch_data[ch]
        d["deg"] = deg
        d["raw"] = raw
        d["agc"] = agc
        d["mag"] = mag
        d["status"] = status
        d["online"] = True

        alpha = self.sensor_smooth_slider.value() / 100.0
        d["smoothed"] = alpha * deg + (1 - alpha) * d["smoothed"]

        w = self._ch_widgets.get(ch)
        if not w:
            return

        delta = self._calc_channel_delta(ch)
        w["angle"].setText(f"{delta:+.1f}°")

        color_map = {"OK": "#0f0", "WEAK": "#ff0",
                     "STRONG": "#f80", "NONE": "#f00"}
        color = color_map.get(status, "#f00")

        mono = cfg.get("gui.font_monospace", "Consolas")
        w["angle"].setStyleSheet(
            f"font-family: {mono}; font-size: 12px; "
            f"background-color: #1a1a2e; color: {color}; "
            f"padding: 2px 4px;")
        w["status"].setStyleSheet(
            f"color: {color}; font-size: 12px;")
        w["status"].setToolTip(
            f"RAW:{raw} AGC:{agc} MAG:{mag} {status}")

        w["raw"].setText(str(raw))
        w["agc"].setText(str(agc))
        w["mag"].setText(status[:3])

    def _on_sensor_error(self, ch: int, error: str):
        if ch in self._ch_data:
            self._ch_data[ch]["online"] = False
            self._ch_data[ch]["status"] = error
        w = self._ch_widgets.get(ch)
        if w:
            w["angle"].setText(" ERR ")
            w["status"].setStyleSheet(
                "color: #f00; font-size: 12px;")
            w["status"].setToolTip(error)

    def _sensor_scan(self):
        if self.sensor_thread:
            self.sensor_thread.send_command("SCAN")

    def _set_channel_zero(self, ch: int):
        if ch in self._ch_data:
            d = self._ch_data[ch]
            d["zero"] = d["deg"]
            d["smoothed"] = d["deg"]
            logger.add(
                f"[Sensors] CH{ch} ноль: {d['zero']:.1f}°")

    def _set_all_zeros(self):
        for ch in self._ch_data:
            if self._ch_data[ch]["online"]:
                self._set_channel_zero(ch)
        logger.add("[Sensors] Все нули установлены")

    def _calc_channel_delta(self, ch: int) -> float:
        d = self._ch_data.get(ch)
        if d is None:
            return 0.0
        ch_cfg = cfg.sensor_channel_cfg(ch)
        scale = ch_cfg.get("scale", 1.0)
        inv = ch_cfg.get("invert", False)
        direction = -1.0 if inv else 1.0
        delta = d["smoothed"] - d["zero"]
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta * scale * direction

    def _apply_sensors_to_joints(self):
        if not self.robot or not self.robot.is_connected:
            return
        if not self.chk_sensor_enable.isChecked():
            return
        if not self.sensor_connected:
            return

        joints_cfg = cfg.get("joints", [])
        joints = list(self.robot.get_joint_positions())
        changed = False

        for ch, d in self._ch_data.items():
            if not d["online"]:
                continue
            w = self._ch_widgets.get(ch)
            if w and not w["check"].isChecked():
                continue
            ch_cfg = cfg.sensor_channel_cfg(ch)
            if not ch_cfg.get("enabled", True):
                continue

            target = ch_cfg.get("target", "joint")
            delta_deg = self._calc_channel_delta(ch)

            if target == "joint":
                ji = ch_cfg.get("joint_index", ch)
                if 0 <= ji < len(joints):
                    if ji < len(joints_cfg):
                        jc = joints_cfg[ji]
                        mn = jc.get("min_deg", -360)
                        mx = jc.get("max_deg", 360)
                        delta_deg = max(mn, min(mx, delta_deg))
                    joints[ji] = math.radians(delta_deg)
                    changed = True

            elif target == "gripper":
                cr = cfg.get("gripper.close_rad", 1.3)
                cd = math.degrees(cr)
                gv = abs(delta_deg) / cd if cd > 0 else 0.0
                gv = max(0.0, min(1.0, gv))
                if cfg.get("gripper.gui_invert", False):
                    gv = 1.0 - gv
                try:
                    self.robot.set_gripper(gv)
                    sp = int(gv * 100)
                    self.gripper_slider.blockSignals(True)
                    self.gripper_slider.setValue(sp)
                    self.gripper_slider.blockSignals(False)
                    self.gripper_label.setText(f"{sp}%")
                except Exception as e:
                    logger.add(f"[Sensors] Ошибка схвата: {e}")

        if changed:
            try:
                self.robot.move_j(joints, blocking=False)
            except Exception as e:
                logger.add(f"[Sensors] Ошибка: {e}")

    # ══════════════════════════════════════════════════════════
    #  Общие
    # ══════════════════════════════════════════════════════════
    def _safe_call(self, func):
        if not self.robot or not self.robot.is_connected:
            logger.add("Робот не подключён!")
            return
        try:
            func()
        except Exception as e:
            logger.add(f"Ошибка: {e}")

    def _on_btn_off(self):
        self._safe_call(lambda: self.robot.move_to_home())

    def move_axis(self, axis: int, direction: int):
        """Шаг задаётся в градусах (step_spin = 5 → 5°)."""
        if not self.robot or not self.robot.is_connected:
            return

        # ── Шаг в ГРАДУСАХ, конвертируем в радианы ──
        step_deg = self.step_spin.value()          # 5 → 5°
        step_rad = math.radians(step_deg)          # 5° → 0.0873 рад

        try:
            if self.move_mode_combo.currentText().startswith(
                    "MoveJ"):
                joints = list(self.robot.get_joint_positions())
                jc = cfg.get("joints", [])

                edir = direction
                if axis < len(jc):
                    if jc[axis].get("gui_invert", False):
                        edir = -direction

                joints[axis] += edir * step_rad

                # Ограничение по лимитам
                if axis < len(jc):
                    mn = math.radians(
                        jc[axis].get("min_deg", -360))
                    mx = math.radians(
                        jc[axis].get("max_deg", 360))
                    joints[axis] = max(mn, min(mx, joints[axis]))

                self.robot.move_j(joints, blocking=False)
            else:
                pose = list(self.robot.get_cartesian_pose())
                # Для линейного: шаг в метрах (mm)
                step_m = step_deg / 1000.0  # 5 → 0.005 м = 5 мм
                pose[axis] += direction * step_m
                self.robot.move_l(pose, blocking=False)
        except Exception as e:
            logger.add(f"Ошибка: {e}")

    # ══════════════════════════════════════════════════════════
    #  Таймер
    # ══════════════════════════════════════════════════════════
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_info)
        self.timer.start(
            cfg.get("application.update_interval_ms", 200))

    def update_info(self):
        if not self.robot or not self.robot.is_connected:
            return
        try:
            cart = self.robot.get_cartesian_pose()
            for i, v in enumerate(cart):
                self.pos_labels[i].setText(f"{v:+.3f}")

            is_j = self.move_mode_combo.currentText() \
                       .startswith("MoveJ")
            jc = cfg.get("joints", [])

            if is_j:
                joints = self.robot.get_joint_positions()
                for i, v in enumerate(joints):
                    deg = np.rad2deg(v)
                    if (i < len(jc)
                            and jc[i].get("gui_invert", False)):
                        deg = -deg
                    self.joy_val_labels[i].setText(f"{deg:.1f}°")
            else:
                for i, v in enumerate(cart):
                    self.joy_val_labels[i].setText(f"{v:.3f}")

            if self.torque_group.isVisible():
                torques = self.robot.get_joint_torques()
                for i, v in enumerate(torques):
                    self.torque_labels[i].setText(f"{v:+.3f}")

            self._apply_sensors_to_joints()

            state = self.robot.state
            sm = {
                RobotState.IDLE:
                    ("ГОТОВ", "green", "white"),
                RobotState.MOVING:
                    ("ДВИЖЕНИЕ", "#2196F3", "white"),
                RobotState.PAUSED:
                    ("ПАУЗА", "yellow", "black"),
                RobotState.EMERGENCY:
                    ("⛔ СТОП", "red", "white"),
                RobotState.ERROR:
                    ("ОШИБКА", "#ff5722", "white"),
            }
            txt, bg, fg = sm.get(state, ("?", "gray", "white"))
            self.status_label.setText(txt)
            self.status_label.setStyleSheet(
                f"background-color: {bg}; color: {fg}; "
                f"font-size: 22px; font-weight: bold; "
                f"padding: 15px;")

            logs = "\n".join(logger.get_logs())
            self.log_view.setText(logs)
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception as e:
            logger.add(f"Ошибка: {e}")

    def save_logs(self):
        logger.set_save_path(self.path_edit.text())
        if logger.save_to_file():
            QMessageBox.information(
                self, "OK", "Логи сохранены!")
        else:
            QMessageBox.critical(
                self, "Ошибка", "Не удалось сохранить")

    def closeEvent(self, event):
        self._stop_sim_render()
        self.stop_camera()
        self.disconnect_sensor()
        if self.robot:
            self.robot.disconnect()
        logger.save_to_file()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotControlGUI()
    window.show()
    sys.exit(app.exec_())