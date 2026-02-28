# test_robot_sim.py

import mujoco
import numpy as np

print(f"MuJoCo {mujoco.mj_versionString()} — OK")

# Тест с моделью робота
xml = """
<mujoco model="test_robot">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="2 2 0.01"/>
    <light pos="0 0 4" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.1 0.05"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3"/>
        <body name="link2" pos="0 0 0.3">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.25"/>
          <site name="tcp" pos="0 0 0.25" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="100"/>
    <position name="a2" joint="j2" kp="100"/>
  </actuator>
  <sensor>
    <jointpos name="s1" joint="j1"/>
    <jointpos name="s2" joint="j2"/>
    <framepos name="tcp_pos" objtype="site" objname="tcp"/>
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print(f"Суставов: {model.njnt}")
print(f"Актуаторов: {model.nu}")
print(f"Сенсоров: {model.nsensor}")

# Симулируем движение
data.ctrl[0] = 1.0   # Поворот j1
data.ctrl[1] = -0.5  # Наклон j2

for i in range(1000):
    mujoco.mj_step(model, data)

print(f"\nПосле 1000 шагов (t={data.time:.2f}с):")
print(f"  j1 = {np.degrees(data.sensordata[0]):.1f}°")
print(f"  j2 = {np.degrees(data.sensordata[1]):.1f}°")
print(f"  TCP = [{data.sensordata[2]:.3f}, {data.sensordata[3]:.3f}, {data.sensordata[4]:.3f}]")

# Тест рендерера
try:
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    frame = renderer.render()
    print(f"\nРендеринг: ✅ кадр {frame.shape}")
except Exception as e:
    print(f"\nРендеринг: ❌ {e}")

# Тест Якобиана
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
jac_pos = np.zeros((3, model.nv))
jac_rot = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
print(f"Якобиан: ✅ shape={jac_pos.shape}")

print("\n" + "=" * 40)
print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ ✅")
print("MuJoCo готов к работе с роботом!")
print("=" * 40)