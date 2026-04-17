import argparse
import csv
import time

import numpy as np
import zmq


FLAT_COUNT = 49
CMD_DIM = 21
TS_BYTES = 8
FLAT_BYTES = FLAT_COUNT * 8
PAYLOAD_BYTES = TS_BYTES + FLAT_BYTES


def q0_ref(t: float) -> float:
    """Joint-0 position reference (rad); must match q_cmd[0] law in main loop."""
    return -0.2 + 0.025 * np.cos(14.0 * t) + 0.08 * np.sin(14.0 * t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe low_state from Raspberry Pi and publish "
            "q_cmd(float64[21]) at fixed rate."
        )
    )
    parser.add_argument(
        "--pi-ip",
        default="192.168.6.132",
        help="Raspberry Pi LAN IP (default: 192.168.6.132). Overwritten if provided.",
    )
    parser.add_argument("--port", type=int, default=5555, help="State publisher port on Raspberry Pi")
    parser.add_argument("--topic", default="low_state", help="State topic name without trailing space")
    parser.add_argument("--cmd-port", type=int, default=5556, help="Command subscriber port on Raspberry Pi")
    parser.add_argument("--cmd-topic", default="low_cmd", help="Command topic name without trailing space")
    parser.add_argument("--cmd-amp", type=float, default=0.3, help="Amplitude A for q0=A*sin(2*pi*f*t)")
    parser.add_argument("--cmd-freq", type=float, default=0.5, help="Frequency f(Hz) for q0=A*sin(2*pi*f*t)")
    parser.add_argument("--cmd-rate", type=float, default=50.0, help="Publish rate for q_cmd in Hz")
    parser.add_argument(
        "--csv-path",
        default="robot_state_export.csv",
        help="Output CSV: q0_cmd, q0_pos (logging starts at first valid low_state, t=0).",
    )
    parser.add_argument(
        "--conflate",
        action="store_true",
        help="Keep only newest message (recommended for real-time control).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sub_endpoint = f"tcp://{args.pi_ip}:{args.port}"
    pub_bind_endpoint = f"tcp://*:{args.cmd_port}"

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    if args.conflate:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(sub_endpoint)

    pub = ctx.socket(zmq.PUB)
    pub.bind(pub_bind_endpoint)

    topic_prefix = (args.topic + " ").encode("ascii")
    sub.setsockopt(zmq.SUBSCRIBE, topic_prefix)
    cmd_topic_prefix = (args.cmd_topic + " ").encode("ascii")

    print(f"[sub] connect={sub_endpoint} topic={args.topic}")
    print("[sub] protocol: topic + space + ts(float64) + flat49(float64[49])")
    print(f"[pub] bind={pub_bind_endpoint} topic={args.cmd_topic} dim={CMD_DIM}")
    print("[pub] protocol: topic + space + ts(float64) + q_cmd(float64[21])")
    print(
        f"[pub] q_cmd rule: q[0]={args.cmd_amp}*sin(2*pi*{args.cmd_freq}*t), "
        "q[1:21]=0"
    )

    header = ["q0_cmd", "q0_pos"]
    csv_file = open(args.csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    print(f"[sub] csv export enabled: {args.csv_path}")
    print("[sub] acquisition: wait for first low_state, then t_cmd=0 and publish starts.")

    sample_idx = 0
    # True after first valid low_state: trajectory time and CSV both start here.
    acq_started = False
    # Latest published q_cmd[0]; paired with each received Pi state row.
    last_q0_cmd = float("nan")
    pub_period = 1.0 / max(args.cmd_rate, 1e-6)
    # Discrete command time: +pub_period per published frame (stable phase; decoupled from wall jitter).
    t_cmd = 0.0
    next_pub_t = time.perf_counter()
    try:
        while True:
            now_perf = time.perf_counter()
            if acq_started and now_perf >= next_pub_t:
                t = t_cmd
                q_cmd = np.zeros(CMD_DIM, dtype=np.float64)
                q_cmd[3] = 0.02+0.005 * np.cos(6.0 * t) - 0.004 * np.sin(6.0 * t)
                q_cmd[6] = 0.04-0.0095 * np.cos(6.0 * t) + 0.005 * np.sin(6.0 * t)
                q_cmd[11] = 0.44-0.037 * np.cos(6.0 * t) + 0.013 * np.sin(6.0 * t)
                q_cmd[15] = -0.425-0.0039 * np.cos(6.0 * t) - 0.023 * np.sin(6.0 * t)-0.0062 * np.cos(28.0 * t) - 0.016 * np.sin(28.0 * t)
                q_cmd[19] = -0.026+0.052 * np.cos(6.0 * t) - 0.014 * np.sin(6.0 * t)-0.014 * np.sin(28.0 * t) + 0.0147 * np.cos(28.0 * t)
                ts_bytes = np.array([time.time()], dtype="<f8").tobytes()
                cmd_payload = ts_bytes + q_cmd.astype("<f8", copy=False).tobytes()
                pub.send(cmd_topic_prefix + cmd_payload)
                last_q0_cmd = float(q_cmd[0])
                t_cmd += pub_period
                next_pub_t += pub_period
                if now_perf > next_pub_t:
                    next_pub_t = now_perf + pub_period
            
            if not sub.poll(timeout=1):
                continue

            msg = sub.recv()
            if not msg.startswith(topic_prefix):
                continue

            payload = msg[len(topic_prefix) :]
            if len(payload) != PAYLOAD_BYTES:
                print(
                    f"[warn] bad payload size: got={len(payload)} expected={PAYLOAD_BYTES}. skip."
                )
                continue

            # Strictly decode as little-endian float64 stream.
            # This matches current C++ sender on common ARM/x86 Linux targets.
            ts = np.frombuffer(payload, dtype="<f8", count=1, offset=0)[0]
            flat = np.frombuffer(payload, dtype="<f8", count=FLAT_COUNT, offset=TS_BYTES)

            q = flat[0:21]
            qd = flat[21:42]
            base_ang_vel = flat[42:45]
            if not acq_started:
                acq_started = True
                t_cmd = 0.0
                next_pub_t = time.perf_counter()
                last_q0_cmd = q0_ref(0.0)
                csv_writer.writerow([f"{last_q0_cmd:.6f}", f"{float(q[0]):.6f}"])
                print("[sub] first low_state: aligned t=0, CSV + publish enabled.")
            else:
                csv_writer.writerow([f"{last_q0_cmd:.6f}", f"{float(q[0]):.6f}"])

            latency_ms = (time.time() - float(ts)) * 1000.0
            sample_idx += 1
            print(
                f"[{sample_idx}] latency={latency_ms:7.1f} ms | "
                f"q0_cmd={last_q0_cmd: .4f} q0_pos={q[0]: .4f} | "
                f"qd0={qd[0]: .4f} w={base_ang_vel}"
            )
    except KeyboardInterrupt:
        print("\n[sub] stopped by user")
    finally:
        csv_file.close()
        sub.close(0)
        pub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
