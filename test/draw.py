import numpy as np
import matplotlib.pyplot as plt

# q_cmd[3] = 0.02 + 0.005 cos(14 t) - 0.004 sin(14 t)  （与 laptop_angletest 中注释一致时可对齐）
t = np.linspace(0.0, 2.0, 2000)
q_cmd3 = 0.02 + 0.005 * np.cos(14.0 * t) - 0.004 * np.sin(14.0 * t)
# 离散差分：dq/dt ≈ np.gradient(q, t)，内部为变步长下的中心差分（端点为一侧差分）
qdot_num = np.gradient(q_cmd3, t)
# 解析：d/dt[0.005 cos(ωt) - 0.004 sin(ωt)] = -0.005 ω sin(ωt) - 0.004 ω cos(ωt)
omega = 14.0
qdot_analytic = -0.005 * omega * np.sin(omega * t) - 0.004 * omega * np.cos(omega * t)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax0.plot(t, q_cmd3, color="C0", lw=1.2)
ax0.set_ylabel(r"$q_{\mathrm{cmd}}[3]$ (rad)")
ax0.set_title(r"$q_{\mathrm{cmd}}[3] = 0.02 + 0.005\cos(14t) - 0.004\sin(14t)$")
ax0.grid(True, alpha=0.3)
ax0.axhline(0.02, color="gray", ls="--", lw=0.8, alpha=0.7, label="offset $0.02$")
ax0.legend(loc="upper right")

ax1.plot(t, qdot_num, color="C1", lw=1.2, label=r"$\mathrm{d}q/\mathrm{d}t$ (np.gradient)")
ax1.plot(t, qdot_analytic, color="k", ls="--", lw=0.9, alpha=0.55, label=r"analytic $\dot q$")
ax1.set_xlabel(r"$t$ (s)")
ax1.set_ylabel(r"$\dot q_{\mathrm{cmd}}[3]$ (rad/s)")
ax1.set_title("Velocity from differentiation of position")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

plt.tight_layout()
plt.show()
