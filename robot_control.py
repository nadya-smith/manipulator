# robot_gui.py

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QLineEdit, QGridLayout, QGroupBox,
                             QComboBox, QSpinBox, QFileDialog, QMessageBox, QProgressBar,
                             QCheckBox, QSlider, QDoubleSpinBox)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from mcx_wrapper import RobotMCX
from logger import logger


# ═══════════════════════════════════════════════════════════════
#  Поток захвата и обработки видео
# ═══════════════════════════════════════════════════════════════
class CameraThread(QThread):
    """Отдельный поток для захвата кадров и детекции объектов."""

    frame_ready = pyqtSignal(np.ndarray)          # Кадр с наложенной графикой
    detection_info = pyqtSignal(list)              # Список найденных объектов

    # ── Диапазоны HSV для каждого цвета ──────────────────────
    # Красный разбит на два диапазона (обёртка H‑канала)
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

    # BGR‑цвета для отрисовки контуров
    DRAW_COLORS = {
        "Красный":  (0,   0,   255),
        "Зелёный":  (0,   255, 0),
        "Синий":    (255, 0,   0),
        "Жёлтый":   (0,   255, 255),
    }

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False
        self.min_area = 500          # Минимальная площадь контура (пиксели²)
        self.enabled_colors = {c: True for c in self.COLOR_RANGES}
        self.show_contours = True
        self.show_bbox = True
        self.cap = None

    # ── Управление ───────────────────────────────────────────
    def set_camera(self, index: int):
        self.camera_index = index

    def set_min_area(self, area: int):
        self.min_area = area

    def set_color_enabled(self, color_name: str, enabled: bool):
        if color_name in self.enabled_colors:
            self.enabled_colors[color_name] = enabled

    # ── Основной цикл ────────────────────────────────────────
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.add(f"[Камера] Не удалось открыть камеру #{self.camera_index}")
            self.running = False
            return

        logger.add(f"[Камера] Подключена камера #{self.camera_index}")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            processed, detections = self._process_frame(frame)
            self.frame_ready.emit(processed)
            if detections:
                self.detection_info.emit(detections)

            self.msleep(30)  # ~33 FPS

        self.cap.release()
        logger.add("[Камера] Отключена")

    def stop(self):
        self.running = False
        self.wait()

    # ── Обработка кадра ──────────────────────────────────────
    def _process_frame(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Лёгкое размытие для уменьшения шума
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        detections = []   # [{"color", "area", "cx", "cy", "contour"}, ...]
        overlay = frame.copy()

        for color_name, ranges in self.COLOR_RANGES.items():
            if not self.enabled_colors.get(color_name, False):
                continue

            # Объединяем маски (для красного — два диапазона)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # Морфология: убираем мелкий шум
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            draw_color = self.DRAW_COLORS[color_name]

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                # Центр масс
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                detections.append({
                    "color": color_name,
                    "area": int(area),
                    "cx": cx,
                    "cy": cy,
                })

                # --- Рисуем контур ---
                if self.show_contours:
                    cv2.drawContours(overlay, [cnt], -1, draw_color, 2)

                    # Полупрозрачная заливка
                    fill = overlay.copy()
                    cv2.drawContours(fill, [cnt], -1, draw_color, cv2.FILLED)
                    cv2.addWeighted(fill, 0.25, overlay, 0.75, 0, overlay)

                # --- Bounding box ---
                if self.show_bbox:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), draw_color, 2)

                # --- Подпись: цвет + площадь ---
                label = f"{color_name}: {area:.0f} px²"  # Исправлено на px²
                # Фон подписи
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(overlay, (cx - 5, cy - th - 8), (cx + tw + 5, cy + 5),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(overlay, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2, cv2.LINE_AA)

                # Маркер центра
                cv2.circle(overlay, (cx, cy), 5, draw_color, -1)
                cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), 1)

        # Общая информация на кадре
        info = f"Объектов: {len(detections)}"
        cv2.putText(overlay, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay, detections


# ═══════════════════════════════════════════════════════════════
#  Главное окно
# ═══════════════════════════════════════════════════════════════
class RobotControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robot = RobotMCX()
        self.robot.connect("COM39")

        self.setWindowTitle("Управление коллаборативным роботом MCX — Модуль А")
        self.setGeometry(100, 100, 1500, 950)

        # Камера
        self.camera_thread = None

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ════════════════════════════════════════════════
        #  ЛЕВАЯ ПАНЕЛЬ: управление роботом
        # ════════════════════════════════════════════════
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 2)

        # ── Кнопки управления ─────────────────────────
        ctrl_group = QGroupBox("Управление роботом")
        ctrl_layout = QHBoxLayout()
        self.btn_on = QPushButton("ВКЛ")
        self.btn_off = QPushButton("ВЫКЛ")
        self.btn_pause = QPushButton("ПАУЗА")
        self.btn_emergency = QPushButton("ЭКСТРЕННОЕ ТОРМОЖЕНИЕ")
        for btn in [self.btn_on, self.btn_off, self.btn_pause, self.btn_emergency]:
            btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.btn_on.setStyleSheet(
            self.btn_on.styleSheet() + "background-color: #4CAF50; color: white;")
        self.btn_emergency.setStyleSheet(
            self.btn_emergency.styleSheet() + "background-color: #f44336; color: white;")
        ctrl_layout.addWidget(self.btn_on)
        ctrl_layout.addWidget(self.btn_off)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_emergency)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)

        self.btn_on.clicked.connect(lambda: logger.add("Робот включён"))
        self.btn_off.clicked.connect(
            lambda: [logger.add("Робот выключен"), self.robot.move_to_start()])
        self.btn_pause.clicked.connect(lambda: logger.add("Робот на паузе"))
        self.btn_emergency.clicked.connect(self.robot.emergency_stop)

        # ── Джойстик ─────────────────────────────────
        joy_group = QGroupBox("Ручное управление (Джойстик)")
        joy_layout = QGridLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["MoveJ (по суставам)", "MoveL (линейно)"])
        joy_layout.addWidget(QLabel("Режим:"), 0, 0)
        joy_layout.addWidget(self.mode_combo, 0, 1, 1, 3)

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
                "background-color: #eee; border: 1px solid #ccc; font-weight: bold;")
            val_lbl.setFixedWidth(70)
            self.joy_val_labels.append(val_lbl)

            btn_plus = QPushButton("+")
            btn_plus.setFixedWidth(40)

            btn_minus.clicked.connect(lambda _, idx=i: self.move_axis(idx, -1))
            btn_plus.clicked.connect(lambda _, idx=i: self.move_axis(idx, 1))

            joy_layout.addWidget(lbl_name,   i + 2, 0)
            joy_layout.addWidget(btn_minus,  i + 2, 1)
            joy_layout.addWidget(val_lbl,    i + 2, 2)
            joy_layout.addWidget(btn_plus,   i + 2, 3)

            self.joint_buttons[i] = (btn_minus, btn_plus)

        joy_group.setLayout(joy_layout)
        left_panel.addWidget(joy_group)

        # ── Схват ─────────────────────────────────────
        grip_group = QGroupBox("Управление схватом")
        grip_layout = QHBoxLayout()
        btn_grip_on = QPushButton("Открыть")
        btn_grip_off = QPushButton("Закрыть")
        btn_grip_on.clicked.connect(lambda: self.robot.set_gripper(1))
        btn_grip_off.clicked.connect(lambda: self.robot.set_gripper(0))
        grip_layout.addWidget(btn_grip_on)
        grip_layout.addWidget(btn_grip_off)
        grip_group.setLayout(grip_layout)
        left_panel.addWidget(grip_group)

        # ── Статус «светофор» ─────────────────────────
        self.status_label = QLabel("ОЖИДАЕТ")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: blue; color: white; font-size: 24px; "
            "font-weight: bold; padding: 20px;")
        left_panel.addWidget(self.status_label)

        # ════════════════════════════════════════════════
        #  ПРАВАЯ ПАНЕЛЬ: информация + камера
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

        # ── Температура (опционально) ─────────────────
        self.show_temp = False
        if self.show_temp:
            temp_group = QGroupBox("Температура моторов")
            temp_layout = QGridLayout()
            self.temp_bars = []
            for i in range(6):
                temp_layout.addWidget(QLabel(f"Мотор {i + 1}:"), i, 0)
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(25 + i * 5)
                temp_layout.addWidget(bar, i, 1)
                self.temp_bars.append(bar)
            temp_group.setLayout(temp_layout)
            right_panel.addWidget(temp_group)

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
        video_group = QGroupBox("Видеопоток с камеры — детекция объектов")
        video_layout = QVBoxLayout()

        # --- Панель настроек камеры ---
        cam_settings = QHBoxLayout()

        # Выбор индекса камеры
        cam_settings.addWidget(QLabel("Камера:"))
        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 10)
        self.cam_index_spin.setValue(0)
        cam_settings.addWidget(self.cam_index_spin)

        # Кнопки старт / стоп
        self.btn_cam_start = QPushButton("▶ Подключить")
        self.btn_cam_start.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_cam_start.clicked.connect(self.start_camera)
        cam_settings.addWidget(self.btn_cam_start)

        self.btn_cam_stop = QPushButton("■ Отключить")
        self.btn_cam_stop.setStyleSheet(
            "background-color: #757575; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_cam_stop.clicked.connect(self.stop_camera)
        self.btn_cam_stop.setEnabled(False)
        cam_settings.addWidget(self.btn_cam_stop)

        # Снимок
        self.btn_snapshot = QPushButton("📷 Снимок")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.btn_snapshot.setEnabled(False)
        cam_settings.addWidget(self.btn_snapshot)

        cam_settings.addStretch()
        video_layout.addLayout(cam_settings)

        # --- Настройки детекции ---
        det_settings = QHBoxLayout()

        det_settings.addWidget(QLabel("Мин. площадь:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(50, 50000)
        self.min_area_spin.setValue(500)
        self.min_area_spin.setSingleStep(100)
        self.min_area_spin.setSuffix(" px²")
        self.min_area_spin.valueChanged.connect(self._on_min_area_changed)
        det_settings.addWidget(self.min_area_spin)

        # Чекбоксы фильтров по цветам
        self.color_checks = {}
        for color_name, draw_clr in CameraThread.DRAW_COLORS.items():
            cb = QCheckBox(color_name)
            cb.setChecked(True)
            # Подкрашиваем текст
            r, g, b = draw_clr[2], draw_clr[1], draw_clr[0]  # BGR → RGB
            cb.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold;")
            cb.stateChanged.connect(
                lambda state, cn=color_name: self._on_color_toggle(cn, state))
            det_settings.addWidget(cb)
            self.color_checks[color_name] = cb

        det_settings.addStretch()
        video_layout.addLayout(det_settings)

        # --- Виджет отображения видео ---
        self.video_label = QLabel("Камера не подключена")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #1a1a2e; color: #aaa; font-size: 16px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # --- Информация о детекции ---
        self.detection_label = QLabel("Обнаруженные объекты: —")
        self.detection_label.setStyleSheet(
            "font-family: Consolas; font-size: 12px; padding: 4px;")
        self.detection_label.setWordWrap(True)
        video_layout.addWidget(self.detection_label)

        video_group.setLayout(video_layout)
        right_panel.addWidget(video_group)

    # ══════════════════════════════════════════════════════════
    #  Камера: старт / стоп / настройки
    # ══════════════════════════════════════════════════════════
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()

        idx = self.cam_index_spin.value()
        self.camera_thread = CameraThread(camera_index=idx, parent=self)
        self.camera_thread.min_area = self.min_area_spin.value()

        # Применяем текущее состояние чекбоксов
        for color_name, cb in self.color_checks.items():
            self.camera_thread.set_color_enabled(color_name, cb.isChecked())

        self.camera_thread.frame_ready.connect(self._display_frame)
        self.camera_thread.detection_info.connect(self._update_detection_info)
        self.camera_thread.start()

        self.btn_cam_start.setEnabled(False)
        self.btn_cam_stop.setEnabled(True)
        self.btn_snapshot.setEnabled(True)
        logger.add(f"[Камера] Запуск потока (индекс {idx})")

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        self.btn_cam_start.setEnabled(True)
        self.btn_cam_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.video_label.setText("Камера отключена")
        self.video_label.setPixmap(QPixmap())          # Очищаем
        self.detection_label.setText("Обнаруженные объекты: —")

    def take_snapshot(self):
        """Сохраняет текущий кадр в файл."""
        pixmap = self.video_label.pixmap()
        if pixmap and not pixmap.isNull():
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить снимок", "snapshot.png",
                "Images (*.png *.jpg *.bmp)")
            if path:
                pixmap.save(path)
                logger.add(f"[Камера] Снимок сохранён: {path}")

    def _on_min_area_changed(self, value):
        if self.camera_thread:
            self.camera_thread.set_min_area(value)

    def _on_color_toggle(self, color_name, state):
        if self.camera_thread:
            self.camera_thread.set_color_enabled(color_name, state == Qt.Checked)

    # ── Отображение кадра ────────────────────────────────────
    def _display_frame(self, frame: np.ndarray):
        """Конвертирует OpenCV‑кадр (BGR) → QPixmap и показывает в QLabel."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def _update_detection_info(self, detections: list):
        """Обновляет текстовую метку с информацией о найденных объектах."""
        if not detections:
            self.detection_label.setText("Обнаруженные объекты: —")
            return

        lines = []
        # Группируем по цвету
        by_color = {}
        for d in detections:
            by_color.setdefault(d["color"], []).append(d)

        for color, items in by_color.items():
            total = sum(it["area"] for it in items)
            lines.append(
                f"  {color}: {len(items)} шт., "
                f"суммарная площадь {total} px²")

        header = f"Объектов: {len(detections)}"
        self.detection_label.setText(header + "\n" + "\n".join(lines))

    # ══════════════════════════════════════════════════════════
    #  Таймер обновления (робот)
    # ══════════════════════════════════════════════════════════
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_info)
        self.timer.start(300)

    def update_info(self):
        try:
            cart = self.robot.get_cart_pos()
            for i, val in enumerate(cart):
                self.pos_labels[i].setText(f"{val:+.3f}")

            is_joint_mode = self.mode_combo.currentText().startswith("MoveJ")
            if is_joint_mode:
                joints = self.robot.get_joint_pos()
                for i, val in enumerate(joints):
                    deg = np.rad2deg(val)
                    self.joy_val_labels[i].setText(f"{deg:.1f}°")
            else:
                for i, val in enumerate(cart):
                    self.joy_val_labels[i].setText(f"{val:.3f}")

            if "ЭКСТРЕННО" in self.log_view.toPlainText():
                self.status_label.setText("АВАРИЙНАЯ ОСТАНОВКА")
                self.status_label.setStyleSheet(
                    "background-color: red; color: white; font-size: 24px; padding: 20px;")
            elif "пауза" in self.log_view.toPlainText().lower():
                self.status_label.setText("ПАУЗА")
                self.status_label.setStyleSheet(
                    "background-color: yellow; color: black; font-size: 24px; padding: 20px;")
            elif self.robot.robot.get_connection():
                self.status_label.setText("В РАБОТЕ")
                self.status_label.setStyleSheet(
                    "background-color: green; color: white; font-size: 24px; padding: 20px;")

            logs = "\n".join(logger.get_logs())
            self.log_view.setText(logs)
            self.log_view.verticalScrollBar().setValue(
                self.log_view.verticalScrollBar().maximum())

        except Exception as e:
            logger.add(f"Ошибка обновления: {e}")

    def move_axis(self, axis: int, direction: int):
        step = self.step_spin.value() / 100.0
        if self.mode_combo.currentText().startswith("MoveJ"):
            joints = list(self.robot.get_joint_pos())
            joints[axis] += direction * step
            self.robot.move_j(joints)
        else:
            pose = list(self.robot.get_cart_pos())
            pose[axis] += direction * step
            self.robot.move_l(pose)

    def save_logs(self):
        logger.set_save_path(self.path_edit.text())
        if logger.save_to_file():
            QMessageBox.information(self, "Успех", "Логи сохранены!")
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить логи")

    def closeEvent(self, event):
        self.stop_camera()
        logger.add("GUI закрыт")
        logger.save_to_file()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotControlGUI()
    window.show()
    sys.exit(app.exec_())