import argparse
import csv
import time
from collections import deque
from typing import Optional

import numpy as np
import zmq


FLAT_COUNT = 49
CMD_DIM = 21
TS_BYTES = 8
FLAT_BYTES = FLAT_COUNT * 8
PAYLOAD_BYTES = TS_BYTES + FLAT_BYTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic host script for ZMQ low_state/low_cmd.\n"
            "Focus: low-overhead command publishing + latency/clock-jump observation."
        )
    )
    parser.add_argument("--pi-ip", default="192.168.1.133", help="Robot IP for low_state subscription")
    parser.add_argument("--port", type=int, default=5555, help="Robot low_state PUB port")
    parser.add_argument("--topic", default="low_state", help="State topic")
    parser.add_argument("--cmd-port", type=int, default=5556, help="Host low_cmd PUB bind port")
    parser.add_argument("--cmd-topic", default="low_cmd", help="Command topic")
    parser.add_argument("--cmd-rate", type=float, default=50.0, help="Command publish rate (Hz)")
    parser.add_argument("--csv-path", default="robot_state_export.csv", help="CSV path when logging enabled")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV writing for pure latency diagnosis")
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print one per N state samples (default: 50; use <=0 to disable sample prints)",
    )
    parser.add_argument(
        "--report-every-sec",
        type=float,
        default=2.0,
        help="Summary report period in seconds",
    )
    parser.add_argument(
        "--latency-window",
        type=int,
        default=500,
        help="Rolling window size for latency statistics",
    )
    parser.add_argument(
        "--clock-jump-threshold-ms",
        type=float,
        default=80.0,
        help=(
            "Warn when wall-vs-monotonic drift changes beyond threshold "
            "(helps catch NTP/system clock step)"
        ),
    )
    parser.add_argument(
        "--conflate",
        dest="conflate",
        action="store_true",
        help="Enable CONFLATE on state SUB (default on)",
    )
    parser.add_argument(
        "--no-conflate",
        dest="conflate",
        action="store_false",
        help="Disable CONFLATE on state SUB",
    )
    parser.set_defaults(conflate=True)
    return parser.parse_args()


def build_cmd(t: float) -> np.ndarray:
    q_cmd = np.zeros(CMD_DIM, dtype=np.float64)
    q_cmd[10] = 0.02 + 0.005 * np.cos(6.0 * t) - 0.004 * np.sin(6.0 * t)
    q_cmd[14] = 0.04 - 0.0095 * np.cos(6.0 * t) + 0.005 * np.sin(6.0 * t)
    q_cmd[18] = 0.44 - 0.037 * np.cos(6.0 * t) + 0.013 * np.sin(6.0 * t)
    return q_cmd


def summarize_latency_ms(vals: deque) -> str:
    if not vals:
        return "n/a"
    arr = np.asarray(vals, dtype=np.float64)
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p99 = np.percentile(arr, 99)
    return f"min={arr.min():.1f} p50={p50:.1f} p90={p90:.1f} p99={p99:.1f} max={arr.max():.1f}"


def main() -> None:
    args = parse_args()
    sub_endpoint = f"tcp://{args.pi_ip}:{args.port}"
    pub_bind_endpoint = f"tcp://*:{args.cmd_port}"
    cmd_period = 1.0 / max(args.cmd_rate, 1e-6)

    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    if args.conflate:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.connect(sub_endpoint)
    topic_prefix = (args.topic + " ").encode("ascii")
    sub.setsockopt(zmq.SUBSCRIBE, topic_prefix)

    pub = ctx.socket(zmq.PUB)
    # Real-time command channel: always latest sample, no close backlog.
    pub.setsockopt(zmq.CONFLATE, 1)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(pub_bind_endpoint)
    cmd_topic_prefix = (args.cmd_topic + " ").encode("ascii")

    csv_file: Optional[object] = None
    csv_writer: Optional[csv.writer] = None
    if not args.no_csv:
        header = [f"q_cmd_{i}" for i in range(CMD_DIM)] + [f"q_pos_{i}" for i in range(CMD_DIM)]
        csv_file = open(args.csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

    print(f"[sub] connect={sub_endpoint} topic={args.topic} conflate={int(args.conflate)}")
    print(f"[pub] bind={pub_bind_endpoint} topic={args.cmd_topic} rate={args.cmd_rate:.2f}Hz")
    print(f"[diag] csv={'off' if args.no_csv else args.csv_path} print-every={args.print_every}")
    print("[diag] waiting first low_state, then command publish starts")

    acq_started = False
    sample_idx = 0
    send_idx = 0
    send_drop_count = 0
    bad_size_count = 0
    last_q_cmd = np.zeros(CMD_DIM, dtype=np.float64)
    lat_hist_ms: deque = deque(maxlen=max(args.latency_window, 10))

    t_cmd = 0.0
    next_pub_perf = time.perf_counter()
    last_report_perf = next_pub_perf

    wall0 = time.time()
    mono0 = next_pub_perf
    drift_prev_ms = 0.0

    try:
        while True:
            now_perf = time.perf_counter()
            now_wall = time.time()

            # Clock-step watcher: drift between wall-clock and monotonic deltas.
            drift_ms = ((now_wall - wall0) - (now_perf - mono0)) * 1000.0
            if abs(drift_ms - drift_prev_ms) > args.clock_jump_threshold_ms:
                print(
                    f"[warn][clock] wall/mono drift jump: prev={drift_prev_ms:.1f}ms "
                    f"now={drift_ms:.1f}ms delta={drift_ms - drift_prev_ms:.1f}ms"
                )
            drift_prev_ms = drift_ms

            if acq_started and now_perf >= next_pub_perf:
                # Publish at most one command per loop; when lagging, drop old schedule ticks.
                q_cmd = build_cmd(t_cmd)
                ts_bytes = np.array([now_wall], dtype="<f8").tobytes()
                payload = ts_bytes + q_cmd.astype("<f8", copy=False).tobytes()
                pub.send(cmd_topic_prefix + payload)
                last_q_cmd = q_cmd
                send_idx += 1

                t_cmd += cmd_period
                next_pub_perf += cmd_period
                if now_perf > next_pub_perf:
                    behind = int((now_perf - next_pub_perf) / cmd_period) + 1
                    send_drop_count += max(behind, 0)
                    next_pub_perf = now_perf + cmd_period

            if sub.poll(timeout=1) == 0:
                now_perf2 = time.perf_counter()
                if now_perf2 - last_report_perf >= max(args.report_every_sec, 0.1):
                    print(
                        f"[report] state=0 send={send_idx} dropped_sched={send_drop_count} "
                        f"lat_ms({summarize_latency_ms(lat_hist_ms)})"
                    )
                    last_report_perf = now_perf2
                continue

            msg = sub.recv()
            if not msg.startswith(topic_prefix):
                continue
            payload = msg[len(topic_prefix) :]
            if len(payload) != PAYLOAD_BYTES:
                bad_size_count += 1
                if bad_size_count <= 5 or bad_size_count % 20 == 0:
                    print(
                        f"[warn] bad payload size: got={len(payload)} expected={PAYLOAD_BYTES} "
                        f"bad-count={bad_size_count}"
                    )
                continue

            ts = float(np.frombuffer(payload, dtype="<f8", count=1, offset=0)[0])
            flat = np.frombuffer(payload, dtype="<f8", count=FLAT_COUNT, offset=TS_BYTES)
            q = flat[0:21]
            qd = flat[21:42]

            if not acq_started:
                acq_started = True
                t_cmd = 0.0
                next_pub_perf = time.perf_counter()
                print("[sub] first low_state received: command publish enabled.")

            if csv_writer is not None:
                # Write raw float values to minimize Python formatting overhead.
                csv_writer.writerow(np.concatenate((last_q_cmd, q)).tolist())

            latency_ms = (time.time() - ts) * 1000.0
            lat_hist_ms.append(latency_ms)
            sample_idx += 1

            if args.print_every > 0 and (sample_idx <= 3 or sample_idx % args.print_every == 0):
                print(
                    f"[{sample_idx}] lat={latency_ms:7.1f}ms qd0={qd[0]: .4f} "
                    f"send={send_idx} dropped_sched={send_drop_count}"
                )

            now_perf3 = time.perf_counter()
            if now_perf3 - last_report_perf >= max(args.report_every_sec, 0.1):
                print(
                    f"[report] state={sample_idx} send={send_idx} dropped_sched={send_drop_count} "
                    f"lat_ms({summarize_latency_ms(lat_hist_ms)})"
                )
                last_report_perf = now_perf3

    except KeyboardInterrupt:
        print("\n[diag] stopped by user")
    finally:
        if csv_file is not None:
            csv_file.close()
        sub.close(0)
        pub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
