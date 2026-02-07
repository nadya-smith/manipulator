# main.py

import os
os.environ["MUJOCO_GL"] = "egl"  # Безопасный рендеринг

import sys
from PyQt5.QtWidgets import QApplication
from robot_control import RobotControlGUI
from logger import logger


def main():
    logger.add("Запуск GUI")
    app = QApplication(sys.argv)
    window = RobotControlGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()