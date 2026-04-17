# MuJoCo 航向保持外环 PD 改动说明（MyRobot）

## 1. 背景与问题

在 `cmd = {vx > 0, vy = 0, vyaw = 0}` 的直行任务中，机器人出现了长期“走弯/侧身走”现象。  
这类问题在强化学习部署中很常见，通常表现为：

- 局部角速度控制正常（`wz` 接近 0）
- 绝对航向角（yaw）仍缓慢漂移
- 长时间积分后轨迹形成弧线或大圈

根因通常是：策略主要在做“瞬时稳定”，并不天然保证“全局朝向保持”。因此在部署侧叠加一个航向外环是成熟工程做法。

---

## 2. 本次改动目标

在不重训策略的前提下，给 `rl_sim_mujoco` 增加“航向保持外环 PD”：

- 使用 IMU 四元数计算当前绝对 yaw
- 维护目标 yaw（`yaw_target`）
- 基于 yaw 误差和 `wz` 计算外环补偿
- 将补偿叠加到原始 `cmd_yaw`，再喂给 policy

---

## 3. 数学原理

### 3.1 误差定义

- 当前航向：`yaw_current`（由 `base_quat` 计算）
- 目标航向：`yaw_target`
- 航向误差：

`yaw_err = wrapToPi(yaw_target - yaw_current)`

`wrapToPi` 的作用是把角度误差限制在 `[-pi, pi]`，避免跨 `+pi/-pi` 时突变。

### 3.2 外环控制律

采用 PD：

`yaw_correction = Kp * yaw_err - Kd * wz`

其中：

- `Kp`：把当前朝向拉回目标朝向
- `Kd`：利用角速度阻尼，抑制振荡
- `wz`：IMU z 轴角速度（`obs.ang_vel[2]`）

### 3.3 输出限幅与融合

- 先对补偿限幅：

`yaw_correction = clamp(yaw_correction, -max_correction, +max_correction)`

- 再与用户输入融合并限幅：

`cmd_yaw_final = clamp(cmd_yaw_user + yaw_correction, -cmd_limit, +cmd_limit)`

这样可以避免外环过度干预导致策略动作突变。

---

## 4. 代码实现（已落地）

## 4.1 工具函数：角度包裹

文件：`src/rl_sar/src/rl_sim_mujoco.cpp`

- 新增 `WrapToPi(float angle)`：
  - 把角度包裹到 `[-pi, pi]`
  - 用于稳定计算 yaw 误差

## 4.2 外环状态变量

文件：`src/rl_sar/include/rl_sim_mujoco.hpp`

- 新增成员：
  - `bool yaw_hold_target_initialized`
  - `float yaw_hold_target`

用途：

- 记录外环目标航向
- 在非 locomotion/暂停时复位，避免跨状态错误继承

## 4.3 主逻辑注入点

文件：`src/rl_sar/src/rl_sim_mujoco.cpp`，函数：`RL_Sim::RunModel()`

外环执行顺序：

1. 读取 `obs.ang_vel` 与 `obs.base_quat`
2. 从四元数计算 `yaw_current`
3. 根据配置读取 `Kp/Kd/限幅/死区阈值`
4. 当用户主动转向或前进速度过小，重置 `yaw_target`
5. 当直行且非主动转向时，计算 PD 补偿并叠加到 `cmd_yaw`
6. 把 `cmd_yaw_final` 写回 `obs.commands`

关键保护逻辑：

- 用户主动转向旁路：`abs(control.yaw) > user_cmd_deadband`
- 低速不控制：`abs(control.x) <= forward_cmd_min`
- 外环关闭时自动复位内部状态

---

## 5. 配置参数（已加入）

文件：`policy/myrobot/robot_lab/config.yaml`

新增字段：

- `yaw_hold_enable: true`
- `yaw_hold_kp: 0.6`
- `yaw_hold_kd: 0.08`
- `yaw_hold_max_correction: 0.3`
- `yaw_hold_cmd_limit: 0.5`
- `yaw_hold_user_cmd_deadband: 0.05`
- `yaw_hold_forward_cmd_min: 0.05`

参数含义：

- `enable`：总开关
- `kp/kd`：PD 增益
- `max_correction`：外环单独输出限幅
- `cmd_limit`：最终下发到 policy 的 yaw 命令限幅
- `user_cmd_deadband`：判断“用户是否主动转向”
- `forward_cmd_min`：低速时不保持航向，避免站立抖动

---

## 6. 工程行为说明

### 6.1 何时锁定目标航向

在以下情况会把 `yaw_target = yaw_current`：

- 外环首次启用
- 用户主动给 yaw 指令
- 前进速度低于阈值（非直行状态）

这保证了“用户手动转向后，松手时从当前朝向继续保持”。

### 6.2 与原策略关系

外环只改 `commands[2]`（yaw 指令通道），不改：

- 观测结构维度
- 网络模型与权重
- action 维度与后处理流程

属于“部署侧高层补偿”，风险和侵入性都较低。

---

## 7. 编译与运行

已验证编译通过：

- `rl_sim_mujoco`
- `rl_sim_mujoco_test`

常用命令：

```bash
cmake --build cmake_build --target rl_sim_mujoco -j4
cmake --build cmake_build --target rl_sim_mujoco_test -j4
```

---

## 8. 初步效果（基于日志对比）

对比日志：

- 旧版本（无外环）：`myrobot_scene_20260408_104648.csv`
- 新版本（有外环）：`myrobot_scene_20260408_135125.csv`

观察到：

- `|yaw_drift|` 峰值从约 `1.77 rad` 降到约 `0.23 rad`
- 新版本中未出现 `|yaw_drift| > 0.3 rad` 的持续偏航
- 直行时 `cmd_yaw` 会出现小幅自动补偿，轨迹显著更直

结论：外环 PD 对“x-only 命令下持续走弯”问题有效。

---

## 9. 调参建议

推荐顺序：

1. 先固定 `Kd`，微调 `Kp`（0.5 -> 0.8）
2. 若出现左右摆，增加 `Kd`（0.08 -> 0.12）
3. 若纠偏过猛，降低 `max_correction`（0.3 -> 0.2~0.25）
4. 若用户转向被抢控制，增大 `user_cmd_deadband`

经验原则：

- 漂移慢但不振荡：优先加 `Kp`
- 有轻微蛇形：优先加 `Kd`
- 有突兀扭头：先降 `max_correction`

---

## 10. 后续可选增强

- 在测试 CSV 中增加：
  - `yaw_err`
  - `yaw_correction`
  - `cmd_yaw_final`
- 增加“只在 `|cmd_x|` 持续高于阈值 N 帧后启用”逻辑
- 若后续要上真机，可再加入轻量 I 项（先小范围验证防积分饱和）

