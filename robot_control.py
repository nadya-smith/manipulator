# robot_control.py

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QLineEdit, QGridLayout, QGroupBox,
                             QComboBox, QSpinBox, QFileDialog, QMessageBox, QProgressBar,
                             QCheckBox, QSlider, QDoubleSpinBox, QFrame)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont

from robot_backend import RobotFactory, RobotMode, RobotState, BaseRobot
from logger import logger


# ═══════════════════════════════════════════════════════════════
#  Поток захвата и обработки видео (с поддержкой виртуальной камеры)
# ═══════════════════════════════════════════════════════════════
class CameraThread(QThread):
    """Захват кадров: с реальной камеры ИЛИ из MuJoCo рендера."""

    frame_ready = pyqtSignal(np.ndarray)
    detection_info = pyqtSignal(list)

    COLOR_RANGES = {
        "Красный": [
            (np.array([0,   100, 100]), np.array([10,  255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255])),
        ],
        "Зелёный": [
            (np.array([35,  80,  80]),  np.array([85,  255, 255])),
        ],
        "Синий": [
            (np.array([100, 80,  80]),  np.array([130, 255, 255])),
        ],
        "Жёлтый": [
            (np.array([20,  100, 100]), np.array([35,  255, 255])),
        ],
    }

    DRAW_COLORS = {
        "Красный":  (0,   0,   255),
        "Зелёный":  (0,   255, 0),
        "Синий":    (255, 0,   0),
        "Жёлтый":   (0,   255, 255),
    }

    def __init__(self, camera_index=0, robot: BaseRobot = None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.robot = robot  # Для получения кадров из MuJoCo
        self.running = False
        self.min_area = 500
        self.enabled_colors = {c: True for c in self.COLOR_RANGES}
        self.show_contours = True
        self.show_bbox = True
        self.cap = None
        self.use_virtual_camera = False  # True если берём кадры из робота

    def set_camera(self, index: int):
        self.camera_index = index

    def set_min_area(self, area: int):
        self.min_area = area

    def set_color_enabled(self, color_name: str, enabled: bool):
        if color_name in self.enabled_colors:
            self.enabled_colors[color_name] = enabled

    def set_virtual_mode(self, enabled: bool, robot: BaseRobot = None):
        """Переключение между реальной и виртуальной камерой."""
        self.use_virtual_camera = enabled
        self.robot = robot

    def run(self):
        self.running = True

        if not self.use_virtual_camera:
            # Реальная камера
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.add(f"[Камера] Не удалось открыть камеру #{self.camera_index}")
                self.running = False
                return
            logger.add(f"[Камера] Подключена камера #{self.camera_index}")
        else:
            logger.add("[Камера] Режим виртуальной камеры (MuJoCo)")

        while self.running:
            frame = None

            if self.use_virtual_camera and self.robot:
                # Кадр из MuJoCo
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

            self.msleep(30)

        if self.cap:
            self.cap.release()
        logger.add("[Камера] Отключена")

    def stop(self):
        self.running = False
        self.wait()

    def _process_frame(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        detections = []
        overlay = frame.copy()

        for color_name, ranges in self.COLOR_RANGES.items():
            if not self.enabled_colors.get(color_name, False):
                continue
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            draw_color = self.DRAW_COLORS[color_name]
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
                    "cx": cx, "cy": cy,
                })
                if self.show_contours:
                    cv2.drawContours(overlay, [cnt], -1, draw_color, 2)
                    fill = overlay.copy()
                    cv2.drawContours(fill, [cnt], -1, draw_color, cv2.FILLED)
                    cv2.addWeighted(fill, 0.25, overlay, 0.75, 0, overlay)
                if self.show_bbox:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), draw_color, 2)
                label = f"{color_name}: {area:.0f} px²"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(overlay, (cx-5, cy-th-8), (cx+tw+5, cy+5),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(overlay, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2,
                            cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), 5, draw_color, -1)
                cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), 1)

        info = f"Объектов: {len(detections)}"
        cv2.putText(overlay, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return overlay, detections


# ═══════════════════════════════════════════════════════════════
#  Главное окно
# ═══════════════════════════════════════════════════════════════
class RobotControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robot: BaseRobot = None  # Текущий робот (создаётся при выборе режима)
        self.camera_thread = None

        self.setWindowTitle(
            "Управление коллаборативным роботом — Реальный / MuJoCo")
        self.setGeometry(100, 100, 1500, 1000)

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ════════════════════════════════════════════════
        #  ЛЕВАЯ ПАНЕЛЬ
        # ════════════════════════════════════════════════
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 2)

        # ── Переключение режима ───────────────────────
        mode_group = QGroupBox("⚙ Режим работы")
        mode_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; }")
        mode_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "🤖 Реальный робот (MCX)",
            "🖥 Симуляция (MuJoCo)"
        ])
        self.mode_selector.setStyleSheet("font-size: 13px; padding: 5px;")
        mode_row.addWidget(QLabel("Режим:"))
        mode_row.addWidget(self.mode_selector)
        mode_layout.addLayout(mode_row)

        # Параметры подключения
        conn_row = QHBoxLayout()
        self.conn_target_edit = QLineEdit("COM39")
        self.conn_target_edit.setPlaceholderText(
            "COM-порт или путь к XML-модели")
        conn_row.addWidget(QLabel("Цель:"))
        conn_row.addWidget(self.conn_target_edit)
        mode_layout.addLayout(conn_row)

        # Чекбокс: показывать MuJoCo viewer
        self.chk_mujoco_viewer = QCheckBox("Показать 3D-просмотрщик MuJoCo")
        self.chk_mujoco_viewer.setChecked(True)
        mode_layout.addWidget(self.chk_mujoco_viewer)

        # Кнопки подключения
        conn_btns = QHBoxLayout()
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

        conn_btns.addWidget(self.btn_connect)
        conn_btns.addWidget(self.btn_disconnect)
        mode_layout.addLayout(conn_btns)

        # Индикатор текущего режима
        self.mode_indicator = QLabel("Не подключено")
        self.mode_indicator.setAlignment(Qt.AlignCenter)
        self.mode_indicator.setStyleSheet(
            "background-color: #666; color: white; "
            "font-size: 13px; padding: 8px; border-radius: 4px;")
        mode_layout.addWidget(self.mode_indicator)

        mode_group.setLayout(mode_layout)
        left_panel.addWidget(mode_group)

        # При переключении режима обновляем подсказки
        self.mode_selector.currentIndexChanged.connect(
            self._on_mode_selector_changed)

        # ── Кнопки управления ─────────────────────────
        ctrl_group = QGroupBox("Управление роботом")
        ctrl_layout = QHBoxLayout()
        self.btn_on = QPushButton("ВКЛ")
        self.btn_off = QPushButton("ВЫКЛ")
        self.btn_pause = QPushButton("ПАУЗА")
        self.btn_resume = QPushButton("ПРОДОЛЖИТЬ")
        self.btn_emergency = QPushButton("ЭКСТРЕННОЕ ТОРМОЖЕНИЕ")
        self.btn_home = QPushButton("HOME")

        for btn in [self.btn_on, self.btn_off, self.btn_pause,
                     self.btn_resume, self.btn_emergency, self.btn_home]:
            btn.setStyleSheet("font-weight: bold; padding: 8px;")

        self.btn_on.setStyleSheet(
            self.btn_on.styleSheet() +
            "background-color: #4CAF50; color: white;")
        self.btn_emergency.setStyleSheet(
            self.btn_emergency.styleSheet() +
            "background-color: #f44336; color: white;")
        self.btn_home.setStyleSheet(
            self.btn_home.styleSheet() +
            "background-color: #FF9800; color: white;")

        ctrl_layout.addWidget(self.btn_on)
        ctrl_layout.addWidget(self.btn_off)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_resume)
        ctrl_layout.addWidget(self.btn_emergency)
        ctrl_layout.addWidget(self.btn_home)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)

        self.btn_on.clicked.connect(
            lambda: logger.add("Робот включён"))
        self.btn_off.clicked.connect(self._on_btn_off)
        self.btn_pause.clicked.connect(self._on_btn_pause)
        self.btn_resume.clicked.connect(self._on_btn_resume)
        self.btn_emergency.clicked.connect(self._on_btn_emergency)
        self.btn_home.clicked.connect(self._on_btn_home)

        # ── Джойстик ─────────────────────────────────
        joy_group = QGroupBox("Ручное управление (Джойстик)")
        joy_layout = QGridLayout()

        self.move_mode_combo = QComboBox()
        self.move_mode_combo.addItems(
            ["MoveJ (по суставам)", "MoveL (линейно)"])
        joy_layout.addWidget(QLabel("Режим:"), 0, 0)
        joy_layout.addWidget(self.move_mode_combo, 0, 1, 1, 3)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 50)
        self.step_spin.setValue(5)
        self.step_spin.setSuffix(" (шаг)")
        joy_layout.addWidget(QLabel("Шаг:"), 1, 0)
        joy_layout.addWidget(self.step_spin, 1, 1, 1, 3)

        axes = ["J1/X", "J2/Y", "J3/Z", "J4/Rx", "J5/Ry", "J6/Rz"]
        self.joint_buttons = {}
        self.joy_val_labels = []

        for i, name in enumerate(axes):
            lbl_name = QLabel(name)
            btn_minus = QPushButton("-")
            btn_minus.setFixedWidth(40)
            val_lbl = QLabel("0.00")
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet(
                "background-color: #eee; border: 1px solid #ccc; "
                "font-weight: bold;")
            val_lbl.setFixedWidth(70)
            self.joy_val_labels.append(val_lbl)
            btn_plus = QPushButton("+")
            btn_plus.setFixedWidth(40)

            btn_minus.clicked.connect(
                lambda _, idx=i: self.move_axis(idx, -1))
            btn_plus.clicked.connect(
                lambda _, idx=i: self.move_axis(idx, 1))

            joy_layout.addWidget(lbl_name,  i+2, 0)
            joy_layout.addWidget(btn_minus, i+2, 1)
            joy_layout.addWidget(val_lbl,   i+2, 2)
            joy_layout.addWidget(btn_plus,  i+2, 3)
            self.joint_buttons[i] = (btn_minus, btn_plus)

        joy_group.setLayout(joy_layout)
        left_panel.addWidget(joy_group)

        # ── Схват ─────────────────────────────────────
        grip_group = QGroupBox("Управление схватом")
        grip_layout = QHBoxLayout()
        btn_grip_open = QPushButton("Открыть")
        btn_grip_close = QPushButton("Закрыть")
        btn_grip_open.clicked.connect(
            lambda: self._safe_call(lambda: self.robot.set_gripper(1.0)))
        btn_grip_close.clicked.connect(
            lambda: self._safe_call(lambda: self.robot.set_gripper(0.0)))
        grip_layout.addWidget(btn_grip_open)
        grip_layout.addWidget(btn_grip_close)
        grip_group.setLayout(grip_layout)
        left_panel.addWidget(grip_group)

        # ── Статус «светофор» ─────────────────────────
        self.status_label = QLabel("НЕ ПОДКЛЮЧЕНО")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: gray; color: white; "
            "font-size: 24px; font-weight: bold; padding: 20px;")
        left_panel.addWidget(self.status_label)

        # ════════════════════════════════════════════════
        #  ПРАВАЯ ПАНЕЛЬ
        # ════════════════════════════════════════════════
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 3)

        # ── Текущая поза ──────────────────────────────
        pos_group = QGroupBox("Текущая поза инструмента")
        pos_layout = QGridLayout()
        self.pos_labels = []
        for i, name in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            pos_layout.addWidget(QLabel(f"{name}:"), i, 0)
            lbl = QLabel("0.000")
            lbl.setStyleSheet("font-family: Consolas;")
            pos_layout.addWidget(lbl, i, 1)
            self.pos_labels.append(lbl)
        pos_group.setLayout(pos_layout)
        right_panel.addWidget(pos_group)

        # ── Моменты (только в симуляции) ──────────────
        self.torque_group = QGroupBox("Крутящие моменты (Н·м)")
        torque_layout = QGridLayout()
        self.torque_labels = []
        for i in range(6):
            torque_layout.addWidget(QLabel(f"J{i+1}:"), i, 0)
            lbl = QLabel("0.000")
            lbl.setStyleSheet("font-family: Consolas;")
            torque_layout.addWidget(lbl, i, 1)
            self.torque_labels.append(lbl)
        self.torque_group.setLayout(torque_layout)
        self.torque_group.setVisible(False)  # Скрыто по умолчанию
        right_panel.addWidget(self.torque_group)

        # ── Логи ──────────────────────────────────────
        log_group = QGroupBox("Логи системы")
        log_layout = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        log_layout.addWidget(self.log_view)

        save_layout = QHBoxLayout()
        self.path_edit = QLineEdit("robot_logs.txt")
        btn_save = QPushButton("Сохранить логи")
        btn_save.clicked.connect(self.save_logs)
        save_layout.addWidget(self.path_edit)
        save_layout.addWidget(btn_save)
        log_layout.addLayout(save_layout)
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)

        # ══════════════════════════════════════════════
        #  ВИДЕОПОТОК С КАМЕРЫ + ДЕТЕКЦИЯ
        # ══════════════════════════════════════════════
        video_group = QGroupBox(
            "Видеопоток — реальная камера / MuJoCo рендер")
        video_layout = QVBoxLayout()

        cam_settings = QHBoxLayout()

        cam_settings.addWidget(QLabel("Камера:"))
        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 10)
        self.cam_index_spin.setValue(0)
        cam_settings.addWidget(self.cam_index_spin)

        # Виртуальная камера (MuJoCo)
        self.chk_virtual_cam = QCheckBox("Виртуальная (MuJoCo)")
        self.chk_virtual_cam.setChecked(False)
        cam_settings.addWidget(self.chk_virtual_cam)

        self.btn_cam_start = QPushButton("▶ Подключить")
        self.btn_cam_start.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 6px 14px;")
        self.btn_cam_start.clicked.connect(self.start_camera)
        cam_settings.addWidget(self.btn_cam_start)

        self.btn_cam_stop = QPushButton("■ Отключить")
        self.btn_cam_stop.setStyleSheet(
            "background-color: #757575; color: white; "
            "font-weight: bold; padding: 6px 14px;")
        self.btn_cam_stop.clicked.connect(self.stop_camera)
        self.btn_cam_stop.setEnabled(False)
        cam_settings.addWidget(self.btn_cam_stop)

        self.btn_snapshot = QPushButton("📷 Снимок")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.btn_snapshot.setEnabled(False)
        cam_settings.addWidget(self.btn_snapshot)

        cam_settings.addStretch()
        video_layout.addLayout(cam_settings)

        # Настройки детекции
        det_settings = QHBoxLayout()
        det_settings.addWidget(QLabel("Мин. площадь:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(50, 50000)
        self.min_area_spin.setValue(500)
        self.min_area_spin.setSingleStep(100)
        self.min_area_spin.setSuffix(" px²")
        self.min_area_spin.valueChanged.connect(self._on_min_area_changed)
        det_settings.addWidget(self.min_area_spin)

        self.color_checks = {}
        for color_name, draw_clr in CameraThread.DRAW_COLORS.items():
            cb = QCheckBox(color_name)
            cb.setChecked(True)
            r, g, b = draw_clr[2], draw_clr[1], draw_clr[0]
            cb.setStyleSheet(
                f"color: rgb({r},{g},{b}); font-weight: bold;")
            cb.stateChanged.connect(
                lambda state, cn=color_name: self._on_color_toggle(cn, state))
            det_settings.addWidget(cb)
            self.color_checks[color_name] = cb

        det_settings.addStretch()
        video_layout.addLayout(det_settings)

        self.video_label = QLabel("Камера не подключена")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; font-size: 16px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        self.detection_label = QLabel("Обнаруженные объекты: —")
        self.detection_label.setStyleSheet(
            "font-family: Consolas; font-size: 12px; padding: 4px;")
        self.detection_label.setWordWrap(True)
        video_layout.addWidget(self.detection_label)

        video_group.setLayout(video_layout)
        right_panel.addWidget(video_group)

    # ══════════════════════════════════════════════════════════
    #  Подключение / отключение робота
    # ══════════════════════════════════════════════════════════

    def connect_robot(self):
        """Создаёт робота выбранного типа и подключает."""
        # Если уже подключены — отключаем
        if self.robot and self.robot.is_connected:
            self.disconnect_robot()

        idx = self.mode_selector.currentIndex()
        mode = RobotMode.REAL if idx == 0 else RobotMode.SIMULATION
        target = self.conn_target_edit.text().strip()

        try:
            self.robot = RobotFactory.create(mode)

            kwargs = {}
            if mode == RobotMode.SIMULATION:
                kwargs["show_viewer"] = self.chk_mujoco_viewer.isChecked()

            success = self.robot.connect(target, **kwargs)

            if success:
                self.btn_connect.setEnabled(False)
                self.btn_disconnect.setEnabled(True)
                self.mode_selector.setEnabled(False)

                mode_text = ("🤖 Реальный робот"
                             if mode == RobotMode.REAL
                             else "🖥 MuJoCo симуляция")
                self.mode_indicator.setText(f"✅ {mode_text}")
                self.mode_indicator.setStyleSheet(
                    "background-color: #4CAF50; color: white; "
                    "font-size: 13px; padding: 8px; border-radius: 4px;")

                # Показать/скрыть виджеты специфичные для режима
                self.torque_group.setVisible(mode == RobotMode.SIMULATION)
                self.chk_virtual_cam.setChecked(mode == RobotMode.SIMULATION)

                logger.add(f"Подключено: {mode_text}")
            else:
                QMessageBox.warning(
                    self, "Ошибка",
                    "Не удалось подключиться. Проверьте настройки.")

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка подключения:\n{e}")
            logger.add(f"Ошибка подключения: {e}")

    def disconnect_robot(self):
        """Отключает текущего робота."""
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
            "font-size: 24px; padding: 20px;")

        logger.add("Робот отключён")

    def _on_mode_selector_changed(self, index):
        """При переключении режима обновляем подсказки."""
        if index == 0:
            self.conn_target_edit.setText("COM39")
            self.conn_target_edit.setPlaceholderText("COM-порт (напр. COM39)")
            self.chk_mujoco_viewer.setVisible(False)
        else:
            self.conn_target_edit.setText("")
            self.conn_target_edit.setPlaceholderText(
                "Путь к XML (пусто = встроенная модель)")
            self.chk_mujoco_viewer.setVisible(True)

    # ══════════════════════════════════════════════════════════
    #  Обработчики кнопок
    # ══════════════════════════════════════════════════════════

    def _safe_call(self, func):
        """Обёртка для безопасного вызова метода робота."""
        if not self.robot or not self.robot.is_connected:
            logger.add("Робот не подключён!")
            return
        try:
            func()
        except Exception as e:
            logger.add(f"Ошибка: {e}")

    def _on_btn_off(self):
        self._safe_call(lambda: self.robot.move_to_home())
        logger.add("Робот → Home")

    def _on_btn_pause(self):
        self._safe_call(lambda: self.robot.pause())

    def _on_btn_resume(self):
        self._safe_call(lambda: self.robot.resume())

    def _on_btn_emergency(self):
        self._safe_call(lambda: self.robot.emergency_stop())

    def _on_btn_home(self):
        self._safe_call(lambda: self.robot.move_to_home())

    # ══════════════════════════════════════════════════════════
    #  Камера
    # ══════════════════════════════════════════════════════════

    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()

        idx = self.cam_index_spin.value()
        use_virtual = self.chk_virtual_cam.isChecked()

        self.camera_thread = CameraThread(
            camera_index=idx, robot=self.robot, parent=self)
        self.camera_thread.use_virtual_camera = use_virtual
        self.camera_thread.min_area = self.min_area_spin.value()

        for color_name, cb in self.color_checks.items():
            self.camera_thread.set_color_enabled(color_name, cb.isChecked())

        self.camera_thread.frame_ready.connect(self._display_frame)
        self.camera_thread.detection_info.connect(self._update_detection_info)
        self.camera_thread.start()

        self.btn_cam_start.setEnabled(False)
        self.btn_cam_stop.setEnabled(True)
        self.btn_snapshot.setEnabled(True)
        src = "MuJoCo рендер" if use_virtual else f"камера #{idx}"
        logger.add(f"[Камера] Запуск: {src}")

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.btn_cam_start.setEnabled(True)
        self.btn_cam_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.video_label.setText("Камера отключена")
        self.video_label.setPixmap(QPixmap())
        self.detection_label.setText("Обнаруженные объекты: —")

    def take_snapshot(self):
        pixmap = self.video_label.pixmap()
        if pixmap and not pixmap.isNull():
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить снимок", "snapshot.png",
                "Images (*.png *.jpg *.bmp)")
            if path:
                pixmap.save(path)
                logger.add(f"[Камера] Снимок: {path}")

    def _on_min_area_changed(self, value):
        if self.camera_thread:
            self.camera_thread.set_min_area(value)

    def _on_color_toggle(self, color_name, state):
        if self.camera_thread:
            self.camera_thread.set_color_enabled(
                color_name, state == Qt.Checked)

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def _update_detection_info(self, detections: list):
        if not detections:
            self.detection_label.setText("Обнаруженные объекты: —")
            return
        lines = []
        by_color = {}
        for d in detections:
            by_color.setdefault(d["color"], []).append(d)
        for color, items in by_color.items():
            total = sum(it["area"] for it in items)
            lines.append(f"  {color}: {len(items)} шт., площадь {total} px²")
        header = f"Объектов: {len(detections)}"
        self.detection_label.setText(header + "\n" + "\n".join(lines))

    # ══════════════════════════════════════════════════════════
    #  Таймер обновления
    # ══════════════════════════════════════════════════════════

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_info)
        self.timer.start(200)

    def update_info(self):
        if not self.robot or not self.robot.is_connected:
            return

        try:
            # Декартова поза
            cart = self.robot.get_cartesian_pose()
            for i, val in enumerate(cart):
                self.pos_labels[i].setText(f"{val:+.3f}")

            # Значения на джойстике
            is_joint_mode = self.move_mode_combo.currentText().startswith(
                "MoveJ")
            if is_joint_mode:
                joints = self.robot.get_joint_positions()
                for i, val in enumerate(joints):
                    deg = np.rad2deg(val)
                    self.joy_val_labels[i].setText(f"{deg:.1f}°")
            else:
                for i, val in enumerate(cart):
                    self.joy_val_labels[i].setText(f"{val:.3f}")

            # Моменты (если доступны)
            if self.torque_group.isVisible():
                torques = self.robot.get_joint_torques()
                for i, val in enumerate(torques):
                    self.torque_labels[i].setText(f"{val:+.3f}")

            # Статус «светофор»
            state = self.robot.state
            status_map = {
                RobotState.IDLE: ("ГОТОВ", "green", "white"),
                RobotState.MOVING: ("В ДВИЖЕНИИ", "#2196F3", "white"),
                RobotState.PAUSED: ("ПАУЗА", "yellow", "black"),
                RobotState.EMERGENCY: ("АВАРИЙНАЯ ОСТАНОВКА", "red", "white"),
                RobotState.ERROR: ("ОШИБКА", "#ff5722", "white"),
                RobotState.DISCONNECTED: ("НЕ ПОДКЛЮЧЕНО", "gray", "white"),
            }
            text, bg, fg = status_map.get(
                state, ("НЕИЗВЕСТНО", "gray", "white"))
            self.status_label.setText(text)
            self.status_label.setStyleSheet(
                f"background-color: {bg}; color: {fg}; "
                f"font-size: 24px; font-weight: bold; padding: 20px;")

            # Логи
            logs = "\n".join(logger.get_logs())
            self.log_view.setText(logs)
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

        except Exception as e:
            logger.add(f"Ошибка обновления: {e}")

    def move_axis(self, axis: int, direction: int):
        if not self.robot or not self.robot.is_connected:
            logger.add("Робот не подключён!")
            return

        step = self.step_spin.value() / 100.0

        try:
            if self.move_mode_combo.currentText().startswith("MoveJ"):
                joints = list(self.robot.get_joint_positions())
                joints[axis] += direction * step
                self.robot.move_j(joints, blocking=False)
            else:
                pose = list(self.robot.get_cartesian_pose())
                pose[axis] += direction * step
                self.robot.move_l(pose, blocking=False)
        except Exception as e:
            logger.add(f"Ошибка движения: {e}")

    def save_logs(self):
        logger.set_save_path(self.path_edit.text())
        if logger.save_to_file():
            QMessageBox.information(self, "Успех", "Логи сохранены!")
        else:
            QMessageBox.critical(
                self, "Ошибка", "Не удалось сохранить логи")

    def closeEvent(self, event):
        self.stop_camera()
        if self.robot:
            self.robot.disconnect()
        logger.add("GUI закрыт")
        logger.save_to_file()
        super().closeEvent(event)
