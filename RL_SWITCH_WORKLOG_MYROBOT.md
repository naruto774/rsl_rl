# MyRobot RL 状态切换调试工作日志

## 1. 问题现象

- 运行命令：`./cmake_build/bin/rl_sim_mujoco myrobot scene`
- 现象：初始姿态正常；切换到 `RLLocomotion` 时机器人会“抖一下 + 后仰一下”，随后被 policy 快速拉回。
- 判断：更像是**状态切换瞬间的控制不连续**，而不是 policy 稳态能力不足。

---

## 2. 初始定位思路

围绕切换链路检查 4 个点：

1. `fsm_myrobot.hpp` 的 `CheckChange()` / `Enter()`  
2. `fsm.hpp` 的 `FSM::Run()`（`NORMAL -> CHANGE`）  
3. `rl_sim_mujoco.cpp` 的 `RobotControl()`（`GetState -> StateController -> SetCommand`）  
4. `rl_sdk.cpp` 的 `RLFSMState::RLControl()`（从 `output_dof_*_queue` 取 RL 输出）  

关键结论：

- 控制线程频率高于 RL 推理线程（受 `decimation` 影响）；
- 状态切换后到首个 RL 输出入队前，存在短暂“空窗期”；
- 空窗期内如果直接切换控制语义，容易造成力矩冲击。

---

## 3. 关键实验：验证“空窗期”是主因

将 `decimation` 从 `4` 改为 `1` 后：

- 抖动幅度显著降低；
- 恢复更快。

说明冲击与空窗期高度相关。时间尺度近似：

\[
T_{gap} \approx (decimation - 1)\cdot dt
\]

---

## 4. 第一阶段修复：首帧门控（First-frame gating）

### 实施内容

- 切入 `RLLocomotion::Enter()` 时：
  - 开启 `rl_wait_first_frame`
  - 缓存 `hold_q_on_enter`
  - 清空 `output_dof_pos/vel/tau` 队列，避免旧帧残留
- 在 `RLFSMState::RLControl()` 中：
  - 若还没拿到首帧 RL 输出：使用 `hold_q_on_enter + fixed_kp/fixed_kd` 保持姿态
  - 首帧 RL 输出到达后再切到 RL 命令

### 效果

- 跳变明显减小，但仍有轻微不连续。

---

## 5. 第二阶段修复：平滑插值（Blend）

在首帧门控后加入短时平滑过渡：

\[
\alpha=\text{clip}(t/T_{blend},0,1),\quad
s(\alpha)=3\alpha^2-2\alpha^3
\]
\[
q^{cmd}=(1-s)q^{hold}+sq^{rl}
,\quad
\dot q^{cmd}=(1-s)0+s\dot q^{rl}
\]
\[
K_p^{cmd}=(1-s)K_p^{fixed}+sK_p^{rl}
,\quad
K_d^{cmd}=(1-s)K_d^{fixed}+sK_d^{rl}
\]

并增加参数：

- `rl_blend_duration`（默认 `0.2s`，可在 `policy/myrobot/robot_lab/config.yaml` 调整）

### 新现象

- `rl_blend_duration` 越大，“横跨一步”越明显。

---

## 6. 根因复盘：插值两端关节语义不一致

虽然加入了插值，但 `q_hold` 和 `q_rl` 的索引语义不一致：

- `q_hold` 来自切换前（旧 `joint_mapping`）
- `q_rl` 来自切换后（新 `joint_mapping`）

直接逐维插值会把不同物理关节混在同一维上，导致横移/跨步异常。

---

## 7. 最终修复：旧映射到新映射重排后再插值

在 `RLLocomotion::Enter()` 中：

1. `InitRL` 前保存
   - `hold_q_old_order`
   - `old_joint_mapping`
2. `InitRL` 后读取 `new_joint_mapping`
3. 以 actuator id 为桥接，完成 old-order -> new-order 重排：

\[
\text{actuator} = old\_mapping[i_{old}]
\]
\[
i_{new} = \arg\{new\_mapping[i] = actuator\}
\]
\[
hold\_q\_{new}[i_{new}] = hold\_q\_{old}[i_{old}]
\]

4. 后续门控与插值统一使用 `hold_q_new_order`
5. remap 失败则 fallback 并打印 warning

### 最终效果

- 状态切换过程无明显跳变；
- `RLLocomotion` 进入时平滑稳定。

---

## 8. 主要修改文件

- `src/rl_sar/fsm_robot/fsm_myrobot.hpp`
  - `RLFSMStateRLLocomotion::Enter()`：门控初始化、队列清空、旧新 mapping 重排
- `src/rl_sar/library/core/rl_sdk/rl_sdk.hpp`
  - 增加门控与插值状态变量
- `src/rl_sar/library/core/rl_sdk/rl_sdk.cpp`
  - `RLFSMState::RLControl()`：等待首帧、smoothstep 插值、平滑切换逻辑
- `policy/myrobot/robot_lab/config.yaml`
  - 新增/调整 `rl_blend_duration`

---

## 9. 调参建议（后续复用）

- `rl_blend_duration` 推荐区间：`0.12 ~ 0.25`  
  - 太小：可能仍有轻微冲击  
  - 太大：动作响应变“软”
- 先保证 mapping 对齐，再做插值；否则插值越长风险越大。
- 若迁移到其他机器人（如 G1/Go2），优先复用“首帧门控 + mapping 重排 + 短时插值”框架。

---

## 10. 一句话总结

本次问题本质是**切换瞬间的时序空窗 + 关节语义不一致**；最终通过“首帧门控 + 顺序对齐 + smoothstep 插值”解决了状态机到 RL 控制的跳变。

