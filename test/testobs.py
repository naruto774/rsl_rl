import argparse
import time

import numpy as np
import zmq


FLAT_COUNT = 49
TS_BYTES = 8
FLAT_BYTES = FLAT_COUNT * 8
PAYLOAD_BYTES = TS_BYTES + FLAT_BYTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe low_state and print raw obs[0:3] quaternion only."
    )
    parser.add_argument("--pi-ip", default="192.168.1.133", help="Robot IP for low_state subscription")
    parser.add_argument("--port", type=int, default=5555, help="Robot low_state PUB port")
    parser.add_argument("--topic", default="low_state", help="State topic")
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one per N received samples",
    )
    parser.add_argument(
        "--conflate",
        dest="conflate",
        action="store_true",
        help="Enable ZMQ CONFLATE (default on)",
    )
    parser.add_argument(
        "--no-conflate",
        dest="conflate",
        action="store_false",
        help="Disable ZMQ CONFLATE",
    )
    parser.set_defaults(conflate=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = f"tcp://{args.pi_ip}:{args.port}"
    topic_prefix = (args.topic + " ").encode("ascii")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    if args.conflate:
        sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.SUBSCRIBE, topic_prefix)
    sub.connect(endpoint)

    print(f"[sub] connect={endpoint} topic={args.topic} conflate={int(args.conflate)}")
    print("[sub] protocol: topic + ts(float64) + flat49(float64[49])")
    print("[diag] printing raw flat[45:49] as quat_w,quat_x,quat_y,quat_z")

    sample_idx = 0
    try:
        while True:
            if sub.poll(timeout=1000) == 0:
                print("[wait] no low_state received")
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
            qos = flat[0:3]
            ang_vel = flat[42:45]

            sample_idx += 1
            if sample_idx % max(args.print_every, 1) == 0:
                lat_ms = (time.time() - ts) * 1000.0
                print(
                    f"[{sample_idx}] obs[0:3]="
                    f"[{qos[0]:+.8f}, {qos[1]:+.8f}, {qos[2]:+.8f} "
                    f"[{ang_vel[0]:+.8f}, {ang_vel[1]:+.8f}, {ang_vel[2]:+.8f}] "
                    f"lat={lat_ms:.1f}ms"
                )
    except KeyboardInterrupt:
        print("\n[exit] stopped by user")
    finally:
        sub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
