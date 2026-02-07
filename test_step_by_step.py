# test_step_by_step.py

import sys
print(f"Python: {sys.version}")

print("[1] Импорт mujoco...")
import mujoco
print(f"    OK: {mujoco.mj_versionString()}")

import numpy as np

print("[2] Загрузка модели...")
xml = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom type="plane" size="1 1 0.01"/>
    <body pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <site name="tcp" pos="0 0 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="100"/>
  </actuator>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print("    OK")

print("[3] Шаг симуляции...")
data.ctrl[0] = 0.5
for i in range(100):
    mujoco.mj_step(model, data)
print(f"    OK: time={data.time:.3f}")

print("[4] Чтение данных...")
print(f"    qpos={data.qpos[0]:.4f}")
print(f"    site_xpos={data.site_xpos[0]}")

print("[5] Якобиан...")
jac_pos = np.zeros((3, model.nv))
jac_rot = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jac_pos, jac_rot, 0)
print(f"    OK: {jac_pos.shape}")

print("[6] Рендерер (создание)...")
sys.stdout.flush()
try:
    renderer = mujoco.Renderer(model, height=240, width=320)
    print("    OK: создан")
except Exception as e:
    print(f"    ОШИБКА: {e}")
    print("\n=== РЕНДЕРИНГ НЕДОСТУПЕН, НО СИМУЛЯЦИЯ РАБОТАЕТ ===")
    sys.exit(0)

print("[7] Рендеринг кадра...")
sys.stdout.flush()
try:
    renderer.update_scene(data)
    frame = renderer.render()
    print(f"    OK: {frame.shape}")
except Exception as e:
    print(f"    ОШИБКА: {e}")

print("\n=== ВСЁ РАБОТАЕТ ===")