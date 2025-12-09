import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QLineEdit, QGridLayout, QGroupBox,
                             QComboBox, QSpinBox, QFileDialog, QMessageBox, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QColor
from mcx_wrapper import RobotMCX
from logger import logger

class RobotControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robot = RobotMCX()
        self.robot.connect()

        self.setWindowTitle("Управление коллаборативным роботом MCX — Модуль А")
        self.setGeometry(100, 100, 1400, 900)

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === Левая панель: управление ===
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 2)

        # Кнопки управления
        ctrl_group = QGroupBox("Управление роботом")
        ctrl_layout = QHBoxLayout()
        self.btn_on = QPushButton("ВКЛ")
        self.btn_off = QPushButton("ВЫКЛ")
        self.btn_pause = QPushButton("ПАУЗА")
        self.btn_emergency = QPushButton("ЭКСТРЕННОЕ ТОРМОЖЕНИЕ")
        for btn in [self.btn_on, self.btn_off, self.btn_pause, self.btn_emergency]:
            btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.btn_on.setStyleSheet(self.btn_on.styleSheet() + "background-color: #4CAF50; color: white;")
        self.btn_emergency.setStyleSheet(self.btn_emergency.styleSheet() + "background-color: #f44336; color: white;")
        ctrl_layout.addWidget(self.btn_on)
        ctrl_layout.addWidget(self.btn_off)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_emergency)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)

        self.btn_on.clicked.connect(lambda: logger.add("Робот включён"))
        self.btn_off.clicked.connect(lambda: [logger.add("Робот выключен"), self.robot.move_to_start()])
        self.btn_pause.clicked.connect(lambda: logger.add("Робот на паузе"))
        self.btn_emergency.clicked.connect(self.robot.emergency_stop)

        # Джойстик
        joy_group = QGroupBox("Ручное управление (Джойстик)")
        joy_layout = QGridLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["MoveJ (по суставам)", "MoveL (линейно)"])
        joy_layout.addWidget(QLabel("Режим:"), 0, 0, 1, 2)
        joy_layout.addWidget(self.mode_combo, 0, 2, 1, 4)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 50)
        self.step_spin.setValue(5)
        self.step_spin.setSuffix(" (0.05 рад / 0.05 м)")
        joy_layout.addWidget(QLabel("Шаг:"), 1, 0, 1, 2)
        joy_layout.addWidget(self.step_spin, 1, 2, 1, 4)

        axes = ["J1/X", "J2/Y", "J3/Z", "J4/Rx", "J5/Ry", "J6/Rz"]
        self.joint_buttons = {}
        for i, name in enumerate(axes):
            lbl = QLabel(name)
            btn_minus = QPushButton("-")
            btn_plus = QPushButton("+")
            btn_minus.clicked.connect(lambda _, idx=i: self.move_axis(idx, -1))
            btn_plus.clicked.connect(lambda _, idx=i: self.move_axis(idx, 1))
            joy_layout.addWidget(lbl, i+2, 0)
            joy_layout.addWidget(btn_minus, i+2, 1)
            joy_layout.addWidget(btn_plus, i+2, 2)
            self.joint_buttons[i] = (btn_minus, btn_plus)

        joy_group.setLayout(joy_layout)
        left_panel.addWidget(joy_group)

        # Схват
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

        # Статус светофор
        self.status_label = QLabel("ОЖИДАЕТ")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: blue; color: white; font-size: 24px; font-weight: bold; padding: 20px;")
        left_panel.addWidget(self.status_label)

        # === Правая панель: информация ===
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 3)

        # Текущие координаты
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

        # Температура моторов
        temp_group = QGroupBox("Температура моторов")
        temp_layout = QGridLayout()
        self.temp_bars = []
        for i in range(6):
            temp_layout.addWidget(QLabel(f"Мотор {i+1}:"), i, 0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(25 + i*5)
            temp_layout.addWidget(bar, i, 1)
            self.temp_bars.append(bar)
        temp_group.setLayout(temp_layout)
        right_panel.addWidget(temp_group)

        # Логи
        log_group = QGroupBox("Логи системы")
        log_layout = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
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

        # Видеопоток (заготовка)
        video_group = QGroupBox("Видеопоток с камеры (модули В/Г)")
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Камера не подключена")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        right_panel.addWidget(video_group)

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_info)
        self.timer.start(300)  # 300 мс

    def update_info(self):
        try:
            cart = self.robot.get_cart_pos()
            for i, val in enumerate(cart):
                self.pos_labels[i].setText(f"{val:+.3f}")

            # Обновление статуса (пример)
            if "ЭКСТРЕННО" in self.log_view.toPlainText():
                self.status_label.setText("АВАРИЙНАЯ ОСТАНОВКА")
                self.status_label.setStyleSheet("background-color: red; color: white; font-size: 24px; padding: 20px;")
            elif "пауза" in self.log_view.toPlainText().lower():
                self.status_label.setText("ПАУЗА")
                self.status_label.setStyleSheet("background-color: yellow; color: black; font-size: 24px; padding: 20px;")
            elif self.robot.robot.get_connection():
                self.status_label.setText("В РАБОТЕ")
                self.status_label.setStyleSheet("background-color: green; color: white; font-size: 24px; padding: 20px;")

            # Логи
            logs = "\n".join(logger.get_logs())
            self.log_view.setText(logs)
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

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
        logger.add("GUI закрыт")
        logger.save_to_file()
        super().closeEvent(event)