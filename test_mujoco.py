# test_mujoco.py — диагностика MuJoCo

import sys
import os
import platform

print(f"Python: {sys.version}")
print(f"Архитектура: {platform.architecture()}")
print(f"Путь Python: {sys.executable}")
print()

# 1. Проверяем версию mujoco
try:
    import importlib.metadata
    ver = importlib.metadata.version("mujoco")
    print(f"mujoco версия: {ver}")
except Exception:
    print("mujoco НЕ установлен")
    sys.exit(1)

# 2. Где находится пакет
import mujoco as mj_module
mj_path = os.path.dirname(mj_module.__file__)
print(f"Путь пакета: {mj_path}")

# 3. Проверяем plugin директорию
plugin_dir = os.path.join(mj_path, "plugin")
if os.path.exists(plugin_dir):
    print(f"\nПлагины в {plugin_dir}:")
    for f in os.listdir(plugin_dir):
        full = os.path.join(plugin_dir, f)
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        print(f"  {f} ({size} bytes)")
else:
    print("Директория plugin НЕ найдена")

# 4. Проверяем Visual C++ Runtime
print("\nПроверка Visual C++ Runtime:")
vc_paths = [
    r"C:\Windows\System32\msvcp140.dll",
    r"C:\Windows\System32\vcruntime140.dll",
    r"C:\Windows\System32\vcruntime140_1.dll",
]
for p in vc_paths:
    exists = os.path.exists(p)
    print(f"  {os.path.basename(p)}: {'✅ OK' if exists else '❌ ОТСУТСТВУЕТ'}")

# 5. Пробуем загрузить DLL вручную
print("\nПопытка загрузить DLL вручную:")
import ctypes
if os.path.exists(plugin_dir):
    for f in os.listdir(plugin_dir):
        if f.endswith(".dll"):
            dll_path = os.path.join(plugin_dir, f)
            try:
                ctypes.CDLL(dll_path)
                print(f"  {f}: ✅ OK")
            except OSError as e:
                print(f"  {f}: ❌ ОШИБКА — {e}")

# 6. Финальная попытка импорта
print("\nПопытка import mujoco:")
try:
    import mujoco
    print(f"✅ Успешно! Версия: {mujoco.mj_versionString()}")
except Exception as e:
    print(f"❌ Ошибка: {e}")