#!/usr/bin/env python3
import argparse
import time

import zmq

#使用方法：python3 test/zmq_bidir_laptop.py --peer-ip 192.168.1.133
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Laptop side ZMQ bidirectional test.")
    p.add_argument("--peer-ip", default="192.168.1.133", help="Robot IP")
    p.add_argument("--tx-port", type=int, default=6001, help="Laptop PUB bind port")
    p.add_argument("--rx-port", type=int, default=6002, help="Robot PUB port to subscribe")
    p.add_argument("--rate", type=float, default=2.0, help="Send rate in Hz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = zmq.Context()

    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.tx_port}")

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.peer_ip}:{args.rx_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "robot")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    period = 1.0 / max(args.rate, 1e-6)
    print(f"[laptop] PUB bind tcp://*:{args.tx_port} -> send 'laptop 1'")
    print(f"[laptop] SUB connect tcp://{args.peer_ip}:{args.rx_port} <- recv 'robot ...'")

    # Give subscriber time to connect to avoid dropping the first messages.
    time.sleep(0.5)

    seq = 0
    try:
        while True:
            seq += 1
            msg = f"laptop 1 seq={seq} t={time.time():.3f}"
            pub.send_string(msg)
            print(f"[TX] {msg}")

            events = dict(poller.poll(timeout=int(period * 1000)))
            if sub in events and events[sub] == zmq.POLLIN:
                recv_msg = sub.recv_string()
                print(f"[RX] {recv_msg}")
    except KeyboardInterrupt:
        print("\n[laptop] stopped")
    finally:
        sub.close(0)
        pub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()

