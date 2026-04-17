#!/usr/bin/env python3
"""Raspberry Pi 侧：ZMQ REP 回显，用于与 laptop_zmqtest.py 做最小往返时延（RTT）测试。

协议：对端 REQ 发来任意 payload，本端立即原样发回。不解析内容，负载大小对延迟影响可忽略时
可用单字节 ping；若需稍大 payload 可后续再扩展。

运行示例::

    python3 laptop_zmqtest.py --samples 200
"""

from __future__ import annotations

import argparse
import time

import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ZMQ REP echo server for simple RTT test (pair with laptop_zmqtest.py)."
    )
    p.add_argument(
        "--bind",
        default="192.168.6.132",
        help="Bind address .",
    )
    p.add_argument(
        "--port",
        type=int,
        default=5557,
        help="REP bind port (default: 5557; avoid collision with state/cmd ports).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print each echo with local processing time (µs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = f"tcp://{args.bind}:{args.port}"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)

    print(f"[pi_zmqtest] REP bind={endpoint}")
    print("[pi_zmqtest] protocol: REQ -> any bytes -> echo same bytes (RTT measured on laptop)")
    print("[pi_zmqtest] Ctrl+C to stop")

    n = 0
    try:
        while True:
            t0 = time.perf_counter()
            _ = sock.recv()
            t1 = time.perf_counter()
            proc_us = (t1 - t0) * 1e6
            sock.send(_)
            n += 1
            if args.verbose:
                print(f"[echo] n={n} proc={proc_us:.1f} µs len={len(_)}")
    except KeyboardInterrupt:
        print(f"\n[pi_zmqtest] stopped, total echoes={n}")
    finally:
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
