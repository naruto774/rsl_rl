#!/usr/bin/env python3
"""
Estimate closed-loop joint bandwidth in MuJoCo under RL_Sim-style PD control.

This script reproduces the low-level control law in rl_sim_mujoco.cpp:
    tau = kp * (q_des - q) + kd * (dq_des - dq)

It runs a sine sweep for selected joints (default: hip_pitch and ankle_pitch)
while keeping all other joints at default pose, then estimates:
  - gain |Q/Qd|
  - phase lag (deg)
  - -3 dB bandwidth (Hz)
  - low-frequency equivalent delay (ms)

Usage example:
  python test/mujoco_joint_bandwidth_sweep.py \
    --xml src/rl_sar_zoo/myrobot_description/mjcf/myrobot.xml \
    --config policy/myrobot/base.yaml \
    --joints left_hip_pitch left_ankle_pitch \
    --amp 0.03 \
    --f-start 0.2 --f-end 8.0 --f-num 20
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _try_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _extract_bracket_value(text: str, key: str) -> str:
    # Find "key: [ ... ]" with possible multiline content.
    m = re.search(rf"(^|\n)\s*{re.escape(key)}\s*:\s*\[", text)
    if not m:
        raise ValueError(f"Cannot find list key '{key}' in yaml.")
    start = m.end() - 1
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise ValueError(f"List for key '{key}' is not closed.")


def _parse_simple_yaml_fallback(yaml_text: str) -> Dict[str, object]:
    def parse_num(key: str) -> float:
        m = re.search(rf"(^|\n)\s*{re.escape(key)}\s*:\s*([\-0-9.eE]+)", yaml_text)
        if not m:
            raise ValueError(f"Cannot find numeric key '{key}'.")
        return float(m.group(2))

    def parse_float_list(key: str) -> List[float]:
        raw = _extract_bracket_value(yaml_text, key)
        body = raw.strip()[1:-1]
        vals = []
        for token in body.split(","):
            t = token.strip()
            if not t:
                continue
            vals.append(float(t))
        return vals

    def parse_str_list(key: str) -> List[str]:
        raw = _extract_bracket_value(yaml_text, key)
        body = raw.strip()[1:-1]
        vals = []
        for token in body.split(","):
            t = token.strip().strip("'").strip('"')
            if t:
                vals.append(t)
        return vals

    return {
        "dt": parse_num("dt"),
        "decimation": parse_num("decimation"),
        "fixed_kp": parse_float_list("fixed_kp"),
        "fixed_kd": parse_float_list("fixed_kd"),
        "default_dof_pos": parse_float_list("default_dof_pos"),
        "joint_names": parse_str_list("joint_names"),
    }


def load_rl_pd_config(config_path: Path) -> Dict[str, object]:
    text = config_path.read_text(encoding="utf-8")
    yaml_mod = _try_import_yaml()
    if yaml_mod is not None:
        data = yaml_mod.safe_load(text)
        if not isinstance(data, dict) or not data:
            raise ValueError("Invalid yaml structure.")
        root_key = next(iter(data.keys()))
        cfg = data[root_key]
        needed = ["dt", "decimation", "fixed_kp", "fixed_kd", "default_dof_pos", "joint_names"]
        for k in needed:
            if k not in cfg:
                raise ValueError(f"Missing key '{k}' in {config_path}.")
        return {
            "dt": float(cfg["dt"]),
            "decimation": float(cfg["decimation"]),
            "fixed_kp": [float(x) for x in cfg["fixed_kp"]],
            "fixed_kd": [float(x) for x in cfg["fixed_kd"]],
            "default_dof_pos": [float(x) for x in cfg["default_dof_pos"]],
            "joint_names": [str(x) for x in cfg["joint_names"]],
        }

    # Fallback parser for this repo's simple yaml style.
    return _parse_simple_yaml_fallback(text)


def logspace(f_start: float, f_end: float, n: int) -> List[float]:
    if n < 2:
        return [f_start]
    ls = []
    l0 = math.log10(f_start)
    l1 = math.log10(f_end)
    for i in range(n):
        a = i / (n - 1)
        ls.append(10 ** (l0 + a * (l1 - l0)))
    return ls


def wrap_to_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def sine_fit_amp_phase(t: Sequence[float], y: Sequence[float], omega: float) -> Tuple[float, float]:
    n = len(t)
    if n == 0:
        return 0.0, 0.0
    # Remove DC bias first.
    y_mean = sum(y) / n
    a = 0.0
    b = 0.0
    for ti, yi in zip(t, y):
        yc = yi - y_mean
        s = math.sin(omega * ti)
        c = math.cos(omega * ti)
        a += yc * s
        b += yc * c
    a *= 2.0 / n
    b *= 2.0 / n
    amp = math.sqrt(a * a + b * b)
    phase = math.atan2(b, a)
    return amp, phase


def estimate_bandwidth(freqs: Sequence[float], gains: Sequence[float]) -> float:
    if not freqs or not gains:
        return float("nan")
    g0 = gains[0]
    if g0 <= 1e-9:
        return float("nan")
    target = g0 / math.sqrt(2.0)
    for i in range(1, len(freqs)):
        g_prev, g_curr = gains[i - 1], gains[i]
        if g_prev >= target and g_curr <= target:
            f_prev, f_curr = freqs[i - 1], freqs[i]
            if abs(g_curr - g_prev) < 1e-12:
                return f_curr
            ratio = (target - g_prev) / (g_curr - g_prev)
            return f_prev + ratio * (f_curr - f_prev)
    return float("nan")


def median(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def main() -> int:
    parser = argparse.ArgumentParser(description="MuJoCo joint bandwidth sweep (RL_Sim-like PD).")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("src/rl_sar_zoo/myrobot_description/mjcf/myrobot.xml"),
        help="MuJoCo XML path (defaults to myrobot.xml).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("policy/myrobot/base.yaml"),
        help="RL config yaml with dt/decimation/fixed_kp/fixed_kd/default_dof_pos/joint_names.",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["left_hip_pitch", "left_ankle_pitch"],
        help="Joint names to sweep.",
    )
    parser.add_argument("--amp", type=float, default=0.03, help="Sine amplitude in rad.")
    parser.add_argument("--f-start", type=float, default=0.2, help="Sweep start frequency (Hz).")
    parser.add_argument("--f-end", type=float, default=8.0, help="Sweep end frequency (Hz).")
    parser.add_argument("--f-num", type=int, default=18, help="Number of sweep frequency points.")
    parser.add_argument("--settle-cycles", type=float, default=4.0, help="Settling cycles per frequency.")
    parser.add_argument("--measure-cycles", type=float, default=6.0, help="Measurement cycles per frequency.")
    parser.add_argument("--min-segment-sec", type=float, default=1.0, help="Min settle/measure duration per point.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("test/mujoco_joint_bandwidth_sweep_results.csv"),
        help="Output CSV file path.",
    )
    parser.add_argument("--print-topn", type=int, default=8, help="Print first N points per joint.")
    args = parser.parse_args()

    try:
        import mujoco  # type: ignore
    except Exception as e:
        print("ERROR: Python package 'mujoco' is required. Install with: pip install mujoco")
        print(f"Import error: {e}")
        return 1

    cfg = load_rl_pd_config(args.config)
    dt = float(cfg["dt"])
    decimation = int(float(cfg["decimation"]))
    control_dt = dt * decimation

    joint_names_cfg: List[str] = list(cfg["joint_names"])  # type: ignore[arg-type]
    default_q: List[float] = list(cfg["default_dof_pos"])  # type: ignore[arg-type]
    kp_list: List[float] = list(cfg["fixed_kp"])  # type: ignore[arg-type]
    kd_list: List[float] = list(cfg["fixed_kd"])  # type: ignore[arg-type]

    if not (len(joint_names_cfg) == len(default_q) == len(kp_list) == len(kd_list)):
        raise ValueError("Config arrays size mismatch: joint_names/default_dof_pos/fixed_kp/fixed_kd.")

    # Name -> index in RL config order.
    cfg_idx = {name: i for i, name in enumerate(joint_names_cfg)}
    for j in args.joints:
        if j not in cfg_idx:
            raise ValueError(f"Joint '{j}' not found in config joint_names.")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    # MuJoCo IDs and addresses.
    joint_id = {}
    qpos_adr = {}
    qvel_adr = {}
    act_id = {}
    ctrl_min = {}
    ctrl_max = {}
    for j in joint_names_cfg:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        if jid < 0 or aid < 0:
            raise ValueError(f"Joint/actuator '{j}' missing in XML.")
        joint_id[j] = jid
        qpos_adr[j] = int(model.jnt_qposadr[jid])
        qvel_adr[j] = int(model.jnt_dofadr[jid])
        act_id[j] = aid
        ctrl_min[j] = float(model.actuator_ctrlrange[aid][0])
        ctrl_max[j] = float(model.actuator_ctrlrange[aid][1])

    # Match rl_sim_mujoco control frequency by using control_dt schedule.
    mujoco_dt = float(model.opt.timestep)
    if control_dt < mujoco_dt:
        raise ValueError(
            f"control_dt ({control_dt}) < MuJoCo timestep ({mujoco_dt}). "
            "Use bigger dt*decimation or smaller model timestep."
        )
    sim_steps_per_ctrl = max(1, int(round(control_dt / mujoco_dt)))
    eff_control_dt = sim_steps_per_ctrl * mujoco_dt

    freqs = logspace(args.f_start, args.f_end, args.f_num)
    rows: List[Dict[str, float]] = []

    print("=== Sweep setup ===")
    print(f"XML: {args.xml}")
    print(f"Config: {args.config}")
    print(f"MuJoCo dt = {mujoco_dt:.6f} s, RL control_dt = {control_dt:.6f} s, effective_dt = {eff_control_dt:.6f} s")
    print(f"Amplitude = {args.amp:.4f} rad, frequencies = {args.f_start:.3f} -> {args.f_end:.3f} Hz ({args.f_num} pts)")
    print(f"Joints: {', '.join(args.joints)}")

    # Helper: one control update (same style as RL_Sim::SetCommand).
    def apply_pd(q_des: Dict[str, float], dq_des: Dict[str, float]) -> None:
        for j in joint_names_cfg:
            q = float(data.qpos[qpos_adr[j]])
            dq = float(data.qvel[qvel_adr[j]])
            idx = cfg_idx[j]
            tau = kp_list[idx] * (q_des[j] - q) + kd_list[idx] * (dq_des[j] - dq)
            tau = max(ctrl_min[j], min(ctrl_max[j], tau))
            data.ctrl[act_id[j]] = tau

    for target_joint in args.joints:
        print(f"\n=== Sweeping joint: {target_joint} ===")
        for f in freqs:
            omega = 2.0 * math.pi * f
            settle_sec = max(args.min_segment_sec, args.settle_cycles / max(f, 1e-6))
            measure_sec = max(args.min_segment_sec, args.measure_cycles / max(f, 1e-6))
            settle_ctrl_n = max(1, int(round(settle_sec / eff_control_dt)))
            measure_ctrl_n = max(1, int(round(measure_sec / eff_control_dt)))

            # Reset each frequency point.
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

            # Initialize qpos to default pose from RL config.
            for j in joint_names_cfg:
                data.qpos[qpos_adr[j]] = default_q[cfg_idx[j]]
                data.qvel[qvel_adr[j]] = 0.0
            mujoco.mj_forward(model, data)

            t = 0.0
            # Settling stage.
            for _ in range(settle_ctrl_n):
                q_des = {}
                dq_des = {}
                for j in joint_names_cfg:
                    q_des_j = default_q[cfg_idx[j]]
                    dq_des_j = 0.0
                    if j == target_joint:
                        q_des_j += args.amp * math.sin(omega * t)
                        dq_des_j = args.amp * omega * math.cos(omega * t)
                    q_des[j] = q_des_j
                    dq_des[j] = dq_des_j
                apply_pd(q_des, dq_des)
                for _k in range(sim_steps_per_ctrl):
                    mujoco.mj_step(model, data)
                t += eff_control_dt

            # Measurement stage.
            ts: List[float] = []
            qd_hist: List[float] = []
            q_hist: List[float] = []
            for _ in range(measure_ctrl_n):
                q_des = {}
                dq_des = {}
                for j in joint_names_cfg:
                    q_des_j = default_q[cfg_idx[j]]
                    dq_des_j = 0.0
                    if j == target_joint:
                        q_des_j += args.amp * math.sin(omega * t)
                        dq_des_j = args.amp * omega * math.cos(omega * t)
                    q_des[j] = q_des_j
                    dq_des[j] = dq_des_j
                apply_pd(q_des, dq_des)
                for _k in range(sim_steps_per_ctrl):
                    mujoco.mj_step(model, data)

                ts.append(t)
                qd_hist.append(q_des[target_joint])
                q_hist.append(float(data.qpos[qpos_adr[target_joint]]))
                t += eff_control_dt

            amp_in, phase_in = sine_fit_amp_phase(ts, qd_hist, omega)
            amp_out, phase_out = sine_fit_amp_phase(ts, q_hist, omega)
            gain = amp_out / max(amp_in, 1e-12)
            phase_diff = wrap_to_pi(phase_out - phase_in)
            phase_lag_deg = -math.degrees(phase_diff)  # positive => output lags input
            phase_lag_rad = -phase_diff
            delay_ms = 1000.0 * phase_lag_rad / max(omega, 1e-9)

            rows.append(
                {
                    "joint": target_joint,
                    "frequency_hz": f,
                    "gain_mag": gain,
                    "phase_lag_deg": phase_lag_deg,
                    "eq_delay_ms": delay_ms,
                    "amp_in_rad": amp_in,
                    "amp_out_rad": amp_out,
                }
            )

        # Per-joint summary.
        rj = [r for r in rows if r["joint"] == target_joint]
        gains = [r["gain_mag"] for r in rj]
        fvec = [r["frequency_hz"] for r in rj]
        bw = estimate_bandwidth(fvec, gains)
        low_freq_delays = [r["eq_delay_ms"] for r in rj[: min(4, len(rj))]]
        delay_med = median(low_freq_delays)
        print(f"Estimated -3 dB bandwidth: {bw:.3f} Hz")
        print(f"Low-frequency equivalent delay (median first points): {delay_med:.2f} ms")
        print("Sample points:")
        for r in rj[: max(1, args.print_topn)]:
            print(
                f"  f={r['frequency_hz']:.3f}Hz | "
                f"|H|={r['gain_mag']:.3f} | "
                f"lag={r['phase_lag_deg']:.2f}deg | "
                f"tau_eq={r['eq_delay_ms']:.2f}ms"
            )

    # Save full sweep CSV.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "joint",
                "frequency_hz",
                "gain_mag",
                "phase_lag_deg",
                "eq_delay_ms",
                "amp_in_rad",
                "amp_out_rad",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved sweep results to: {args.out_csv}")
    print(
        "NOTE: This is closed-loop bandwidth under RL_Sim-style PD in MuJoCo, "
        "not intrinsic servo bandwidth."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
