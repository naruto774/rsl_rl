#!/usr/bin/env python3
"""
PC-side ZMQ sine sweep for end-to-end joint bandwidth measurement.

Chain under test:
    PC q_cmd -> ZMQ -> Pi actuator execution -> ZMQ low_state -> PC

Protocol (must match rl_real_myrobot.cpp and Pi side):
  - low_state: "topic " + ts(float64) + flat49(float64[49])
      flat[0:21]   = q
      flat[21:42]  = qd
      flat[42:45]  = ang_vel
      flat[45:49]  = quat
  - low_cmd: "topic " + ts(float64) + q_cmd(float64[21])

Notes:
  - This script keeps all joints at 0 by default, and injects sine only on
    selected joints (left_hip_pitch, left_ankle_pitch by default), as requested.
  - It outputs ONE CSV containing the excitation trajectory (cmd_q) together
    with the tracked response (meas_q) for every logged tick across all joints
    and frequencies. settle and measure phases are both logged with continuous
    t_rel_sec so the curves can be plotted as-is.
  - Bandwidth / phase-lag / equivalent-delay summary is printed to stdout only
    (no separate summary CSV).
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


# Protocol constants (same as rl_real_myrobot.hpp)
ZMQ_STATE_DIM = 49
ZMQ_CMD_DIM = 21
ZMQ_TS_BYTES = 8
ZMQ_STATE_PAYLOAD_BYTES = ZMQ_TS_BYTES + ZMQ_STATE_DIM * 8
ZMQ_CMD_PAYLOAD_BYTES = ZMQ_TS_BYTES + ZMQ_CMD_DIM * 8

# robot_lab_real order (policy/myrobot/robot_lab_real/config.yaml)
JOINT_INDEX = {
    "left_hip_pitch": 0,
    "right_hip_pitch": 1,
    "waist_pitch": 2,
    "left_hip_roll": 3,
    "right_hip_roll": 4,
    "waist_yaw": 5,
    "left_hip_yaw": 6,
    "right_hip_yaw": 7,
    "head": 8,
    "left_shoulder_pitch": 9,
    "right_shoulder_pitch": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_shoulder_roll": 13,
    "right_shoulder_roll": 14,
    "left_ankle_pitch": 15,
    "right_ankle_pitch": 16,
    "left_elbow": 17,
    "right_elbow": 18,
    "left_ankle_roll": 19,
    "right_ankle_roll": 20,
}


def logspace(f_start: float, f_end: float, n: int) -> List[float]:
    if n <= 1:
        return [f_start]
    l0 = math.log10(f_start)
    l1 = math.log10(f_end)
    return [10 ** (l0 + (l1 - l0) * (i / (n - 1))) for i in range(n)]


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
    y_mean = sum(y) / n
    a = 0.0
    b = 0.0
    for ti, yi in zip(t, y):
        yc = yi - y_mean
        a += yc * math.sin(omega * ti)
        b += yc * math.cos(omega * ti)
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
    g_target = g0 / math.sqrt(2.0)
    for i in range(1, len(freqs)):
        if gains[i - 1] >= g_target and gains[i] <= g_target:
            f0, f1 = freqs[i - 1], freqs[i]
            g0_, g1_ = gains[i - 1], gains[i]
            if abs(g1_ - g0_) < 1e-12:
                return f1
            a = (g_target - g0_) / (g1_ - g0_)
            return f0 + a * (f1 - f0)
    return float("nan")


def percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    w = idx - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def parse_state_message(msg: bytes, topic: str) -> Dict[str, object] | None:
    prefix = (topic + " ").encode("utf-8")
    if not msg.startswith(prefix):
        return None
    payload = msg[len(prefix) :]
    if len(payload) < ZMQ_STATE_PAYLOAD_BYTES:
        return None

    ts = struct.unpack_from("<d", payload, 0)[0]
    flat = struct.unpack_from("<49d", payload, ZMQ_TS_BYTES)
    q = list(flat[0:21])
    qd = list(flat[21:42])
    ang_vel = list(flat[42:45])
    quat = list(flat[45:49])
    return {"state_ts": ts, "q": q, "qd": qd, "ang_vel": ang_vel, "quat": quat}


def build_cmd_packet(topic: str, cmd_q: Sequence[float], ts: float) -> bytes:
    if len(cmd_q) != ZMQ_CMD_DIM:
        raise ValueError(f"cmd_q length must be {ZMQ_CMD_DIM}")
    prefix = (topic + " ").encode("utf-8")
    payload = struct.pack("<d", ts) + struct.pack("<21d", *cmd_q)
    return prefix + payload


def recv_latest_state(sub: Any, topic: str, zmq_mod: Any) -> Tuple[Dict[str, object] | None, float]:
    latest = None
    t_last_recv = float("nan")
    while True:
        try:
            msg = sub.recv(flags=zmq_mod.NOBLOCK)
        except zmq_mod.Again:
            break
        parsed = parse_state_message(msg, topic)
        if parsed is not None:
            latest = parsed
            t_last_recv = time.perf_counter()
    return latest, t_last_recv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end ZMQ joint bandwidth sweep (PC->Pi->PC).")
    p.add_argument("--pi-ip", default="192.168.6.132", help="Pi IP for low_state SUB connect.")
    p.add_argument("--state-port", type=int, default=5555, help="Pi state PUB port.")
    p.add_argument("--cmd-port", type=int, default=5556, help="PC command PUB bind port.")
    p.add_argument("--state-topic", default="low_state", help="State topic prefix.")
    p.add_argument("--cmd-topic", default="low_cmd", help="Command topic prefix.")
    p.add_argument(
        "--joints",
        nargs="+",
        default=["left_hip_pitch", "left_ankle_pitch"],
        help="Target joints to sweep.",
    )
    p.add_argument("--amp", type=float, default=0.03, help="Sine amplitude (rad).")
    p.add_argument("--f-start", type=float, default=0.2, help="Sweep start frequency (Hz).")
    p.add_argument("--f-end", type=float, default=8.0, help="Sweep end frequency (Hz).")
    p.add_argument("--f-num", type=int, default=18, help="Number of sweep points.")
    p.add_argument("--settle-cycles", type=float, default=4.0, help="Settling cycles per frequency.")
    p.add_argument("--measure-cycles", type=float, default=6.0, help="Measurement cycles per frequency.")
    p.add_argument("--min-segment-sec", type=float, default=1.0, help="Minimum settle/measure duration.")
    p.add_argument("--rate-hz", type=float, default=200.0, help="Command/logging rate (Hz).")
    p.add_argument("--startup-sec", type=float, default=1.5, help="PUB warm-up before sending commands.")
    p.add_argument("--state-timeout-sec", type=float, default=5.0, help="Timeout waiting first state.")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("test/real_zmq_joint_bandwidth_trace.csv"),
        help="Single CSV with excitation (cmd_q) and tracked response (meas_q).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="After sweep, plot cmd vs meas time series and Bode (gain/phase).",
    )
    p.add_argument(
        "--plot-save",
        type=Path,
        default=None,
        help="Save plots to this prefix (e.g. test/bw). If unset, plots are shown interactively.",
    )
    return p.parse_args()


def plot_results(
    csv_path: Path,
    summary_rows: List[Dict[str, float | str]],
    save_prefix: Path | None,
) -> None:
    """Read trace CSV and plot:
       (a) cmd vs meas time series for every (joint, frequency)
       (b) Bode-style gain & phase-lag vs frequency, per joint
    Falls back gracefully if matplotlib is missing.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[plot] matplotlib not available, skip plotting ({e}).")
        print("       install with: pip install matplotlib")
        return

    # Group trace by (joint, freq) -> {t, cmd, meas, phase}
    series: Dict[Tuple[str, float], Dict[str, List]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["joint"], float(row["frequency_hz"]))
            d = series.setdefault(key, {"t": [], "cmd": [], "meas": [], "phase": []})
            d["t"].append(float(row["t_rel_sec"]))
            d["cmd"].append(float(row["cmd_q_rad"]))
            d["meas"].append(float(row["meas_q_rad"]))
            d["phase"].append(row["phase"])

    joints = sorted({k[0] for k in series.keys()})

    # (a) Time-series grid: one figure per joint, subplot per frequency.
    for joint in joints:
        freqs = sorted({k[1] for k in series.keys() if k[0] == joint})
        n = len(freqs)
        if n == 0:
            continue
        ncol = min(4, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.4 * nrow), sharex=False)
        # Normalize axes shape to 2D list.
        if nrow == 1 and ncol == 1:
            ax_list = [axes]
        elif nrow == 1 or ncol == 1:
            ax_list = list(axes).copy() if hasattr(axes, "__iter__") else [axes]
        else:
            ax_list = [a for row in axes for a in row]

        for i, freq in enumerate(freqs):
            ax = ax_list[i]
            d = series[(joint, freq)]
            ax.plot(d["t"], d["cmd"], color="tab:blue", lw=1.0, label="cmd")
            ax.plot(d["t"], d["meas"], color="tab:red", lw=1.0, label="meas")
            # Shade settle region for clarity.
            settle_t = [t for t, ph in zip(d["t"], d["phase"]) if ph == "settle"]
            if settle_t:
                ax.axvspan(min(settle_t), max(settle_t), color="gray", alpha=0.08)
            ax.set_title(f"{freq:.2f} Hz", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8, loc="upper right")
        # Hide unused subplots.
        for j in range(n, len(ax_list)):
            ax_list[j].axis("off")
        fig.suptitle(f"{joint}: excitation vs tracking", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        if save_prefix is not None:
            out = save_prefix.with_name(save_prefix.name + f"_{joint}_trace.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=120)
            print(f"[plot] saved {out}")

    # (b) Bode-style summary plot, one figure for all joints.
    if summary_rows:
        fig, (ax_g, ax_p) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        for joint in joints:
            sj = [r for r in summary_rows if r["joint"] == joint]
            sj.sort(key=lambda r: float(r["frequency_hz"]))
            fj = [float(r["frequency_hz"]) for r in sj]
            gj = [float(r["gain_mag"]) for r in sj]
            pj = [float(r["phase_lag_deg"]) for r in sj]
            ax_g.semilogx(fj, gj, "-o", label=joint, ms=4)
            ax_p.semilogx(fj, pj, "-o", label=joint, ms=4)
        ax_g.axhline(1.0 / math.sqrt(2.0), color="k", ls="--", lw=0.8, label="-3 dB")
        ax_g.set_ylabel("|H(f)| = amp_out / amp_in")
        ax_g.grid(True, which="both", alpha=0.3)
        ax_g.legend(fontsize=8)
        ax_p.set_ylabel("phase lag [deg]")
        ax_p.set_xlabel("frequency [Hz]")
        ax_p.grid(True, which="both", alpha=0.3)
        fig.suptitle("Closed-loop ZMQ end-to-end response", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        if save_prefix is not None:
            out = save_prefix.with_name(save_prefix.name + "_bode.png")
            fig.savefig(out, dpi=120)
            print(f"[plot] saved {out}")

    if save_prefix is None:
        plt.show()


def main() -> int:
    args = parse_args()
    try:
        import zmq  # type: ignore
    except Exception as e:
        print("ERROR: Python package 'pyzmq' is required. Install with: pip install pyzmq")
        print(f"Import error: {e}")
        return 1

    for j in args.joints:
        if j not in JOINT_INDEX:
            raise ValueError(f"Unsupported joint name: {j}")

    f_points = logspace(args.f_start, args.f_end, args.f_num)
    period = 1.0 / max(args.rate_hz, 1e-6)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    pub = ctx.socket(zmq.PUB)

    # Keep latest state only, matching rl_real_myrobot.cpp behavior.
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.SUBSCRIBE, (args.state_topic + " ").encode("utf-8"))
    sub.connect(f"tcp://{args.pi_ip}:{args.state_port}")
    pub.bind(f"tcp://*:{args.cmd_port}")

    print(f"[sweep] SUB connect tcp://{args.pi_ip}:{args.state_port} topic={args.state_topic}")
    print(f"[sweep] PUB bind tcp://*:{args.cmd_port} topic={args.cmd_topic}")
    print(f"[sweep] rate={args.rate_hz:.1f}Hz ({period*1000.0:.2f}ms)")
    print(f"[sweep] joints={args.joints}")
    print(f"[sweep] sweep={args.f_start:.3f}..{args.f_end:.3f}Hz, points={args.f_num}, amp={args.amp:.4f}rad")

    print(f"[sweep] waiting {args.startup_sec:.2f}s for PUB/SUB link warm-up...")
    time.sleep(args.startup_sec)

    # Wait first state.
    t_wait0 = time.perf_counter()
    latest_state = None
    latest_recv_t = float("nan")
    while time.perf_counter() - t_wait0 < args.state_timeout_sec:
        st, tr = recv_latest_state(sub, args.state_topic, zmq)
        if st is not None:
            latest_state = st
            latest_recv_t = tr
            break
        time.sleep(0.002)
    if latest_state is None:
        print("[error] no low_state received. Check Pi publisher/topic/port.")
        pub.close(0)
        sub.close(0)
        ctx.term()
        return 1
    print("[sweep] first low_state received.")

    args.csv.parent.mkdir(parents=True, exist_ok=True)

    trace_f = args.csv.open("w", newline="", encoding="utf-8")
    trace_writer = csv.writer(trace_f)
    # Minimal columns: target excitation vs tracked response, with phase tag
    # so one can either plot the full sweep or filter to the measure window.
    trace_writer.writerow(
        [
            "joint",
            "frequency_hz",
            "phase",          # "settle" or "measure"
            "t_rel_sec",
            "cmd_q_rad",      # excitation trajectory (target)
            "meas_q_rad",     # tracked response
            "meas_qd_radps",
        ]
    )

    summary_rows: List[Dict[str, float | str]] = []

    try:
        for joint in args.joints:
            jidx = JOINT_INDEX[joint]
            print(f"\n=== Sweeping {joint} (idx={jidx}) ===")
            for freq in f_points:
                omega = 2.0 * math.pi * freq
                settle_sec = max(args.min_segment_sec, args.settle_cycles / max(freq, 1e-6))
                meas_sec = max(args.min_segment_sec, args.measure_cycles / max(freq, 1e-6))
                settle_n = max(1, int(round(settle_sec * args.rate_hz)))
                meas_n = max(1, int(round(meas_sec * args.rate_hz)))

                # Keep command vector all zero except current sweep joint.
                cmd_vec = [0.0] * ZMQ_CMD_DIM

                t0 = time.perf_counter()
                next_tick = t0
                # Settling + measurement in one loop to keep phase continuous.
                ts_meas: List[float] = []
                qd_meas: List[float] = []
                q_meas: List[float] = []
                state_age_hist: List[float] = []

                total_n = settle_n + meas_n
                for k in range(total_n):
                    # Pull latest state before sending current command.
                    st, tr = recv_latest_state(sub, args.state_topic, zmq)
                    if st is not None:
                        latest_state = st
                        latest_recv_t = tr

                    t_now = time.perf_counter()
                    t_rel = t_now - t0
                    cmd_val = args.amp * math.sin(omega * t_rel)
                    cmd_vec[jidx] = cmd_val

                    send_ts_epoch = time.time()
                    packet = build_cmd_packet(args.cmd_topic, cmd_vec, send_ts_epoch)
                    pub.send(packet, flags=zmq.NOBLOCK)

                    # Pull state after sending as well (best effort newest sample).
                    st2, tr2 = recv_latest_state(sub, args.state_topic, zmq)
                    if st2 is not None:
                        latest_state = st2
                        latest_recv_t = tr2

                    if latest_state is not None:
                        meas_q = float(latest_state["q"][jidx])  # type: ignore[index]
                        meas_qd = float(latest_state["qd"][jidx])  # type: ignore[index]
                        state_ts = float(latest_state["state_ts"])  # type: ignore[arg-type]
                        is_measure = k >= settle_n

                        # Always log so excitation+tracking curves are continuous.
                        trace_writer.writerow(
                            [
                                joint,
                                freq,
                                "measure" if is_measure else "settle",
                                t_rel,
                                cmd_val,
                                meas_q,
                                meas_qd,
                            ]
                        )

                        if is_measure:
                            state_age = max(0.0, time.time() - state_ts)
                            ts_meas.append(t_rel)
                            qd_meas.append(cmd_val)
                            q_meas.append(meas_q)
                            state_age_hist.append(state_age)

                    next_tick += period
                    sleep_s = next_tick - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        # Overrun: resync to avoid accumulating drift.
                        next_tick = time.perf_counter()

                amp_in, phase_in = sine_fit_amp_phase(ts_meas, qd_meas, omega)
                amp_out, phase_out = sine_fit_amp_phase(ts_meas, q_meas, omega)
                gain = amp_out / max(amp_in, 1e-12)
                phase_diff = wrap_to_pi(phase_out - phase_in)
                phase_lag_deg = -math.degrees(phase_diff)
                eq_delay_ms = 1000.0 * max(0.0, -phase_diff) / max(omega, 1e-9)

                err = [q - qd for q, qd in zip(q_meas, qd_meas)]
                rms_err = math.sqrt(sum(e * e for e in err) / max(len(err), 1))
                p95_abs_err = percentile([abs(e) for e in err], 0.95)
                state_age_med_ms = 1000.0 * percentile(state_age_hist, 0.5) if state_age_hist else float("nan")

                summary_rows.append(
                    {
                        "joint": joint,
                        "frequency_hz": freq,
                        "gain_mag": gain,
                        "phase_lag_deg": phase_lag_deg,
                        "eq_delay_ms": eq_delay_ms,
                        "amp_in_rad": amp_in,
                        "amp_out_rad": amp_out,
                        "rms_tracking_err_rad": rms_err,
                        "p95_abs_tracking_err_rad": p95_abs_err,
                        "state_age_median_ms": state_age_med_ms,
                    }
                )
                print(
                    f"  f={freq:6.3f}Hz | |H|={gain:6.3f} | lag={phase_lag_deg:7.2f}deg | "
                    f"tau_eq={eq_delay_ms:6.2f}ms | rms_e={rms_err:7.4f}rad"
                )

            # Per-joint bandwidth summary (stdout only).
            sj = [r for r in summary_rows if r["joint"] == joint]
            fj = [float(r["frequency_hz"]) for r in sj]
            gj = [float(r["gain_mag"]) for r in sj]
            bw = estimate_bandwidth(fj, gj)
            low_delay = [float(r["eq_delay_ms"]) for r in sj[: min(4, len(sj))]]
            delay_med = percentile(low_delay, 0.5) if low_delay else float("nan")
            print(f"[result] {joint}: estimated -3dB bandwidth = {bw:.3f} Hz, low-freq eq-delay = {delay_med:.2f} ms")

    finally:
        # Send zero command before exit.
        zero_cmd = [0.0] * ZMQ_CMD_DIM
        for _ in range(5):
            try:
                pub.send(build_cmd_packet(args.cmd_topic, zero_cmd, time.time()), flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            time.sleep(0.005)
        trace_f.close()
        pub.close(0)
        sub.close(0)
        ctx.term()

    print(f"\nSaved trace (excitation + tracking): {args.csv}")

    if args.plot:
        plot_results(args.csv, summary_rows, args.plot_save)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
