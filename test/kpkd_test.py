#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kpkd_test.py — MuJoCo 中悬空测试 left_hip_pitch 的 PD 跟踪效果

主要功能
--------
1. 加载 myrobot 的 MJCF 模型（src/rl_sar_zoo/myrobot_description/mjcf/myrobot.xml）。
2. 加载 base.yaml 中的 fixed_kp / fixed_kd / default_dof_pos / joint_names。
3. 关闭重力使机器人悬空（不修改 XML）；其它 20 个关节用 PD 锁在 default_dof_pos，
   只对 left_hip_pitch 注入正弦目标轨迹：
       q_target(t) = q_default[0] + A * sin(2π f t)
4. 控制律（与真机 joint-level PD 一致）：
       τ = kp * (q_target - q) - kd * dq         （再 clip 到 torque_limits）
5. 实时打开 mujoco.viewer 观察，并把
       time, q_target, q_actual, dq_actual, torque
   按 50 Hz（policy 频率）写到 CSV。

运行方式
--------
推荐用已装好 mujoco 的 conda env：
    conda activate unitree    # 或 go2
    python test/kpkd_test.py --amp 0.3 --freq 1.0 --duration 10.0

Sim-to-Real 注意
----------------
base.yaml 的 joint_names 是 IsaacSim 训练顺序，与 MJCF actuator 的物理顺序不同。
脚本通过 mj_name2id 按名字查找，建立 policy_idx → mj_qpos/qvel/actuator 的映射，
保证 kp/kd/target 与真机部署的索引一致。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import mujoco
import mujoco.viewer


# ===== 默认路径（相对仓库根目录）=====
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "src/rl_sar_zoo/myrobot_description/mjcf/myrobot.xml"
DEFAULT_YAML = REPO_ROOT / "policy/myrobot/robot_lab_real/base.yaml"
DEFAULT_CSV = REPO_ROOT / "test/kpkd_trace.csv"
TARGET_JOINT = "left_hip_pitch"


def load_yaml_cfg(yaml_path: Path) -> dict:
    """读取 base.yaml，自动取出唯一的顶层配置块。"""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    # 兼容 "myrobot/robot_lab_real:" 这种唯一 key 的嵌套结构
    if isinstance(raw, dict) and len(raw) == 1:
        cfg = next(iter(raw.values()))
    else:
        cfg = raw
    required = ["fixed_kp", "fixed_kd", "default_dof_pos", "joint_names", "torque_limits"]
    for k in required:
        if k not in cfg:
            raise KeyError(f"base.yaml 缺少字段: {k}")
    return cfg


def build_index_maps(model: mujoco.MjModel, joint_names: list[str]):
    """
    建立 policy_idx -> (qpos_adr, qvel_adr, actuator_id) 映射。
    保证与 base.yaml 中 joint_names 的顺序严格对齐。
    """
    qpos_adr = np.zeros(len(joint_names), dtype=np.int32)
    qvel_adr = np.zeros(len(joint_names), dtype=np.int32)
    act_id = np.zeros(len(joint_names), dtype=np.int32)

    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MJCF 中找不到关节: {name}")
        qpos_adr[i] = model.jnt_qposadr[jid]
        qvel_adr[i] = model.jnt_dofadr[jid]

        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"MJCF 中找不到 actuator: {name}（应与 joint 同名）")
        act_id[i] = aid
    return qpos_adr, qvel_adr, act_id


def main():
    parser = argparse.ArgumentParser(description="悬空测试 left_hip_pitch 的 PD 跟踪效果")
    parser.add_argument("--xml", type=str, default=str(DEFAULT_XML), help="MJCF 文件路径")
    parser.add_argument("--yaml", type=str, default=str(DEFAULT_YAML), help="base.yaml 路径")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="输出 CSV 路径")
    parser.add_argument("--amp", type=float, default=0.3, help="正弦幅值 (rad)")
    parser.add_argument("--freq", type=float, default=1.0, help="正弦频率 (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="测试时长 (s)")
    parser.add_argument("--no-viewer", action="store_true", help="不开 viewer，纯 headless")
    args = parser.parse_args()

    # ---------- 1) 加载模型与配置 ----------
    cfg = load_yaml_cfg(Path(args.yaml))
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    dt_phys = float(cfg.get("dt", 0.005))
    decimation = int(cfg.get("decimation", 4))
    model.opt.timestep = dt_phys
    # 控制频率 = 物理频率 / decimation = 1/(dt_phys*decimation)
    ctrl_dt = dt_phys * decimation

    # 关闭重力实现悬空（不改 XML）
    model.opt.gravity[:] = 0.0

    joint_names = list(cfg["joint_names"])
    kp = np.asarray(cfg["fixed_kp"], dtype=np.float64)
    kd = np.asarray(cfg["fixed_kd"], dtype=np.float64)
    q_default = np.asarray(cfg["default_dof_pos"], dtype=np.float64)
    tau_lim = np.asarray(cfg["torque_limits"], dtype=np.float64)
    n = len(joint_names)
    assert kp.shape == kd.shape == q_default.shape == tau_lim.shape == (n,), \
        "kp/kd/default_dof_pos/torque_limits 长度必须与 joint_names 一致"

    qpos_adr, qvel_adr, act_id = build_index_maps(model, joint_names)

    # 找到目标关节在 policy 顺序中的 index（应为 0，但保险起见用名字查）
    target_idx = joint_names.index(TARGET_JOINT)
    q0_target = q_default[target_idx]

    # ---------- 2) 初始化 qpos 到 default ----------
    # 先 reset 到 keyframe / 默认零位，再把 21 个关节摆到 default_dof_pos
    mujoco.mj_resetData(model, data)
    for i in range(n):
        data.qpos[qpos_adr[i]] = q_default[i]
    # base freejoint 抬高一点，纯粹为了 viewer 视觉好看；重力已关，不会掉
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        # qpos[0:3] 是 base 位置，qpos[3:7] 是四元数 (w, x, y, z)
        data.qpos[2] = max(data.qpos[2], 0.6)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    # ---------- 3) 主循环：物理步 + PD ----------
    n_steps = int(args.duration / dt_phys)
    log_every = decimation  # 50 Hz 采样

    log_rows = []  # time, q_target, q_actual, dq_actual, torque
    print(f"[INFO] 物理步长 dt={dt_phys}s, 控制频率={1/ctrl_dt:.1f}Hz, 总步数={n_steps}")
    print(f"[INFO] 目标关节 {TARGET_JOINT} (policy_idx={target_idx}), "
          f"kp={kp[target_idx]}, kd={kd[target_idx]}, q_default={q0_target}")
    print(f"[INFO] 正弦轨迹: A={args.amp} rad, f={args.freq} Hz")

    viewer_ctx = None
    if not args.no_viewer:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)

    try:
        wall_t0 = time.time()
        for step in range(n_steps):
            t = step * dt_phys

            # 目标轨迹（policy 顺序的 q_target 数组）
            q_target_vec = q_default.copy()
            q_target_vec[target_idx] = q0_target + args.amp * np.sin(2 * np.pi * args.freq * t)

            # 读取实际状态（按 policy 顺序索引）
            q_actual = data.qpos[qpos_adr]
            dq_actual = data.qvel[qvel_adr]

            # joint-level PD：tau = kp*(q* - q) - kd*dq
            tau = kp * (q_target_vec - q_actual) - kd * dq_actual
            tau = np.clip(tau, -tau_lim, tau_lim)

            # 写入 ctrl（actuator 物理顺序）
            data.ctrl[act_id] = tau

            mujoco.mj_step(model, data)

            # 50 Hz 采样：取 step % decimation == 0 时的状态（执行 step 后）
            if step % log_every == 0:
                log_rows.append((
                    t,
                    q_target_vec[target_idx],
                    float(data.qpos[qpos_adr[target_idx]]),
                    float(data.qvel[qvel_adr[target_idx]]),
                    float(tau[target_idx]),
                ))

            # viewer 同步 + 实时性
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    print("[INFO] viewer 已关闭，提前结束")
                    break
                viewer_ctx.sync()
                # 简单实时同步：让 wall-clock 跟 sim-clock 对齐
                wall_target = wall_t0 + (step + 1) * dt_phys
                sleep = wall_target - time.time()
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    # ---------- 4) 保存 CSV + 简要统计 ----------
    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(log_rows, dtype=np.float64)
    header = "time,q_target,q_actual,dq_actual,torque"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"[INFO] CSV 已保存: {out_path} (rows={len(arr)})")

    if len(arr) > 0:
        err = arr[:, 1] - arr[:, 2]
        # 跳过前 0.5s 暂态再统计稳态指标
        mask = arr[:, 0] > 0.5
        rms = float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else float("nan")
        max_abs = float(np.max(np.abs(err[mask]))) if mask.any() else float("nan")
        tau_peak = float(np.max(np.abs(arr[:, 4])))
        print(f"[STAT] 跟踪误差 RMS = {rms:.4f} rad,  Max |err| = {max_abs:.4f} rad")
        print(f"[STAT] 力矩峰值 = {tau_peak:.4f} Nm  (limit = {tau_lim[target_idx]:.3f} Nm)")
        if tau_peak >= 0.95 * tau_lim[target_idx]:
            print("[WARN] 力矩接近/触及上限，PD 可能饱和，增益评估不可靠！")


if __name__ == "__main__":
    main()
