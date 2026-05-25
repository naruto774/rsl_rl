import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

trace_path = Path(__file__).with_name("trace.csv")
df = pd.read_csv(trace_path)
df = df[df["fsm_state"] == "RLFSMStateRLLocomotion"].copy()
df["t_sec"] = df["t_sec"] - df["t_sec"].iloc[0]

t = df["t_sec"]

joint_id = 0
q = df[f"dof_pos_{joint_id}"]
dq = df[f"dof_vel_{joint_id}"]
action = df[f"action_{joint_id}"]

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(t, df["base_ang_vel_x"], label="wx")
plt.plot(t, df["base_ang_vel_y"], label="wy")
plt.plot(t, df["base_ang_vel_z"], label="wz")
plt.title("Base Angular Velocity")
plt.ylabel("rad/s")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, df["base_quat_w"], label="qw")
plt.plot(t, df["base_quat_x"], label="qx")
plt.plot(t, df["base_quat_y"], label="qy")
plt.plot(t, df["base_quat_z"], label="qz")
plt.title("Base Quaternion")
plt.ylabel("quat")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, q, label=f"dof_pos_{joint_id} [rad]")
plt.plot(t, dq, label=f"dof_vel_{joint_id} [rad/s]")
plt.plot(t, action, label=f"action_{joint_id} [raw]")
plt.title(f"Joint {joint_id}: Position, Velocity, Action")
plt.xlabel("time [s]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

joint_id = 0
default_q = -0.3   # 对应 joint 0，按 config 里的 default_dof_pos
action_scale = 0.25

q_des_from_action = default_q + action_scale * df[f"action_{joint_id}"]
q_target_live = df[f"dof_pos_target_{joint_id}"]

plt.figure()
plt.plot(t, q_des_from_action, label="q_des from raw action")
plt.plot(t, q_target_live, label="live q_target to PD")
plt.legend()
plt.grid(True)
plt.show()