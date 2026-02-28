import os
from datetime import datetime
from typing import List

class Logger:
    def __init__(self):
        self.logs: List[str] = []
        self.emergency_logs: List[str] = []
        self.file_path = "robot_logs.txt"

    def add(self, message: str, emergency: bool = False):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        full_msg = f"[{timestamp}] {message}"
        self.logs.append(full_msg)
        if emergency:
            self.emergency_logs.append(full_msg)

    def get_logs(self) -> List[str]:
        return self.logs[-100:]  # последние 100 строк

    def set_save_path(self, path: str):
        self.file_path = path or "robot_logs.txt"

    def save_to_file(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.logs))
            with open("emergency_logs.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(self.emergency_logs))
            return True
        except Exception as e:
            self.add(f"Ошибка сохранения логов: {e}", emergency=True)
            return False

logger = Logger()