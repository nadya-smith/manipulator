import sys
from PyQt5.QtWidgets import QApplication
from robot_control import RobotControlGUI
from logger import logger

if __name__ == "__main__":
    logger.add("Запуск GUI — Модуль А")
    app = QApplication(sys.argv)
    window = RobotControlGUI()
    window.show()
# %%
    sys.exit(app.exec_())
