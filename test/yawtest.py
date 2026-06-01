#!/usr/bin/env python3
import argparse
import csv
import math
import time
from typing import Optional

import numpy as np
import zmq


FLAT_COUNT = 49
TS_BYTES = 8
PAYLOAD_BYTES = TS_BYTES + FLAT_COUNT * 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe low_state and print quat/yaw only (no command publish). "
            "Optionally compare measured quat to sim reference."
        )
    )
    parser.add_argument("--pi-ip", default="192.168.1.133", help="Robot IP for low_state subscription")
    parser.add_argument("--port", type=int, default=5555, help="Robot low_state PUB port")
    parser.add_argument("--topic", default="low_state", help="State topic")
    parser.add_argument("--conflate", action="store_true", help="Enable ZMQ CONFLATE (keep newest only)")
    parser.add_argument("--print-every", type=int, default=20, help="Print one sample per N messages")
    parser.add_argument("--report-every-sec", type=float, default=2.0, help="Summary period (seconds)")
    parser.add_argument(
        "--sim-trace",
        default="/home/elephant/robot/rl_sar/policy/myrobot/trace_sim.csv",
        help="Sim trace path for reference quaternion extraction",
    )
    parser.add_argument(
        "--sim-state",
        default="RLFSMStateRLWholeBodyTrackingDance",
        help="FSM state used to extract reference from sim trace",
    )
    parser.add_argument(
        "--sim-skip",
        type=int,
        default=2,
        help="Skip first N rows in selected sim state (avoid transition frames)",
    )
    parser.add_argument(
        "--sim-window",
        type=int,
        default=20,
        help="Average next N rows after skip to form reference quat",
    )
    parser.add_argument(
        "--ref-quat",
        default="",
        help="Optional manual ref quat as 'w,x,y,z'. If set, overrides sim trace.",
    )
    return parser.parse_args()


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def align_quat_sign_to_anchor(q: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    # q and -q represent the same rotation; enforce sign continuity against anchor.
    if float(np.dot(q, anchor)) < 0.0:
        return -q
    return q


def average_quats_wxyz_sign_aligned(quats: list[np.ndarray]) -> np.ndarray:
    if not quats:
        raise ValueError("quats must be non-empty")
    aligned: list[np.ndarray] = []
    anchor = normalize_quat_wxyz(quats[0])
    aligned.append(anchor)
    for q in quats[1:]:
        qn = normalize_quat_wxyz(q)
        qn = align_quat_sign_to_anchor(qn, anchor)
        aligned.append(qn)
    q_avg = np.mean(np.stack(aligned, axis=0), axis=0)
    return normalize_quat_wxyz(q_avg)


def yaw_deg_from_quat_wxyz(q: np.ndarray) -> float:
    # ZYX yaw from quaternion (w, x, y, z)
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def wrap_deg_180(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_angle_diff_deg(q_ref: np.ndarray, q_cur: np.ndarray) -> float:
    # shortest SO(3) geodesic angle
    q_rel = normalize_quat_wxyz(quat_mul_wxyz(quat_conj_wxyz(q_ref), q_cur))
    w = abs(float(q_rel[0]))
    w = max(-1.0, min(1.0, w))
    return math.degrees(2.0 * math.acos(w))


def rotate_vec_by_quat_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # v' = q * [0,v] * conj(q)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul_wxyz(quat_mul_wxyz(q, qv), quat_conj_wxyz(q))[1:]


def parse_ref_quat_arg(s: str) -> Optional[np.ndarray]:
    if not s.strip():
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--ref-quat must be in format w,x,y,z")
    q = np.array([float(p) for p in parts], dtype=np.float64)
    return normalize_quat_wxyz(q)


def load_ref_quat_from_sim_trace(trace_path: str, state_name: str, skip: int, window: int) -> np.ndarray:
    quats = []
    with open(trace_path, newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            if not row or len(row) < 12:
                continue
            if row[1] != state_name:
                continue
            q = np.array(
                [float(row[8]), float(row[9]), float(row[10]), float(row[11])],
                dtype=np.float64,
            )
            if float(np.linalg.norm(q)) < 1e-8:
                continue
            quats.append(normalize_quat_wxyz(q))

    if not quats:
        raise RuntimeError(f"No valid quaternions found in state '{state_name}' from {trace_path}")

    i0 = max(0, int(skip))
    i1 = min(len(quats), i0 + max(1, int(window)))
    if i0 >= i1:
        i0 = 0
        i1 = min(len(quats), max(1, int(window)))

    # Handle quaternion double-cover explicitly (q and -q are same rotation).
    # Without sign alignment, direct averaging can cancel into near-zero vectors.
    return average_quats_wxyz_sign_aligned(quats[i0:i1])


def main() -> None:
    args = parse_args()
    ref_quat = parse_ref_quat_arg(args.ref_quat)
    ref_src = "manual --ref-quat"
    if ref_quat is None:
        ref_quat = load_ref_quat_from_sim_trace(
            args.sim_trace, args.sim_state, args.sim_skip, args.sim_window
        )
        ref_src = (
            f"{args.sim_trace} state={args.sim_state} "
            f"skip={args.sim_skip} window={args.sim_window}"
        )
    ref_yaw = yaw_deg_from_quat_wxyz(ref_quat)
    ref_fwd = rotate_vec_by_quat_wxyz(ref_quat, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    endpoint = f"tcp://{args.pi_ip}:{args.port}"
    topic_prefix = (args.topic + " ").encode("ascii")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    if args.conflate:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.SUBSCRIBE, topic_prefix)
    sub.connect(endpoint)

    print(f"[sub] connect={endpoint} topic={args.topic} conflate={int(args.conflate)}")
    print("[sub] protocol: topic + ts(float64) + flat49(float64[49])")
    print(f"[ref] source={ref_src}")
    print(
        f"[ref] quat=[{ref_quat[0]:+.4f},{ref_quat[1]:+.4f},{ref_quat[2]:+.4f},{ref_quat[3]:+.4f}] "
        f"yaw={ref_yaw:+7.2f}deg fwd_xy=({ref_fwd[0]:+.3f},{ref_fwd[1]:+.3f})"
    )
    print("[diag] rotate robot by hand until angle_err is minimal (close to 0 deg)")

    sample_idx = 0
    lat_hist = []
    last_report_t = time.perf_counter()
    best_err_deg = float("inf")
    best_sample = -1
    best_yaw = 0.0
    best_quat = ref_quat.copy()
    prev_quat: Optional[np.ndarray] = None

    try:
        while True:
            if sub.poll(timeout=1000) == 0:
                continue

            msg = sub.recv()
            if not msg.startswith(topic_prefix):
                continue
            payload = msg[len(topic_prefix):]
            if len(payload) != PAYLOAD_BYTES:
                print(f"[warn] bad payload size: {len(payload)} != {PAYLOAD_BYTES}")
                continue

            ts = float(np.frombuffer(payload, dtype="<f8", count=1, offset=0)[0])
            flat = np.frombuffer(payload, dtype="<f8", count=FLAT_COUNT, offset=TS_BYTES)
            quat = normalize_quat_wxyz(flat[45:49].astype(np.float64, copy=False))
            if prev_quat is not None:
                quat = align_quat_sign_to_anchor(quat, prev_quat)
            prev_quat = quat
            yaw_deg = yaw_deg_from_quat_wxyz(quat)
            yaw_err_deg = wrap_deg_180(yaw_deg - ref_yaw)
            angle_err_deg = quat_angle_diff_deg(ref_quat, quat)
            fwd = rotate_vec_by_quat_wxyz(quat, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            lat_ms = (time.time() - ts) * 1000.0

            lat_hist.append(lat_ms)
            if len(lat_hist) > 300:
                lat_hist.pop(0)

            sample_idx += 1
            if angle_err_deg < best_err_deg:
                best_err_deg = angle_err_deg
                best_sample = sample_idx
                best_yaw = yaw_deg
                best_quat = quat.copy()

            if args.print_every <= 1 or sample_idx % args.print_every == 0:
                print(
                    f"[{sample_idx}] quat=[{quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f}] "
                    f"yaw={yaw_deg:+7.2f}deg yaw_err={yaw_err_deg:+7.2f}deg "
                    f"angle_err={angle_err_deg:6.2f}deg fwd_xy=({fwd[0]:+.3f},{fwd[1]:+.3f}) "
                    f"lat={lat_ms:7.1f}ms"
                )

            now = time.perf_counter()
            if now - last_report_t >= max(args.report_every_sec, 0.1):
                arr = np.asarray(lat_hist, dtype=np.float64)
                if arr.size > 0:
                    p50 = float(np.percentile(arr, 50))
                    p95 = float(np.percentile(arr, 95))
                    print(
                        f"[report] n={sample_idx} lat(ms): min={arr.min():.1f} "
                        f"p50={p50:.1f} p95={p95:.1f} max={arr.max():.1f} | "
                        f"best_angle_err={best_err_deg:.2f}deg@#{best_sample} best_yaw={best_yaw:+.2f}deg"
                    )
                last_report_t = now
    except KeyboardInterrupt:
        print("\n[diag] stopped by user")
        if best_sample > 0:
            print(
                f"[best] sample={best_sample} angle_err={best_err_deg:.2f}deg "
                f"yaw={best_yaw:+.2f}deg quat=[{best_quat[0]:+.4f},{best_quat[1]:+.4f},"
                f"{best_quat[2]:+.4f},{best_quat[3]:+.4f}]"
            )
    finally:
        sub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
