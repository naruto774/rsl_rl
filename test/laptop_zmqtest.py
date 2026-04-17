#!/usr/bin/env python3
"""笔记本侧：ZMQ REQ 向树莓派发 ping，测量往返时延（RTT）。

数学上，单次样本的 RTT 定义为
\\( \\mathrm{RTT}_i = t^{\\mathrm{recv}}_i - t^{\\mathrm{send}}_i \\)，
其中 \\(t^{\\mathrm{send}}_i\\)、\\(t^{\\mathrm{recv}}_i\\) 为同一机器上的单调时钟
（`time.perf_counter()`）。在链路对称且处理时间可忽略时，粗略单程延迟可估计为 \\(\\mathrm{RTT}/2\\)，
实际跟踪回路还需叠加应用层调度与串行化开销。

需先在 Pi 上运行 :file:`pi_zmqtest.py` 再运行本脚本。

运行示例::

    python3 laptop_zmqtest.py --pi-ip 192.168.6.132 --port 5557 --samples 200 --rate 50
"""

from __future__ import annotations

import argparse
import statistics
import time

import zmq


PING_PAYLOAD = b"p"  # 最小负载；与数据包大小无关的粗测


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ZMQ REQ ping to Raspberry Pi REP echo; measure RTT (pair with pi_zmqtest.py)."
    )
    p.add_argument(
        "--pi-ip",
        default="192.168.6.132",
        help="Raspberry Pi LAN IP.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=5557,
        help="REP port on Pi (must match pi_zmqtest.py).",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of RTT samples after warm-up.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up pings (discarded, default 5).",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Scheduled ping rate in Hz (default 50). send() is aligned to 1/rate seconds.",
    )
    return p.parse_args()


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    w = idx - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def main() -> None:
    args = parse_args()
    rate_hz = max(args.rate, 1e-6)
    period_s = 1.0 / rate_hz
    endpoint = f"tcp://{args.pi_ip}:{args.port}"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.connect(endpoint)

    print(f"[laptop_zmqtest] REQ connect={endpoint}")
    print(
        f"[laptop_zmqtest] payload_len={len(PING_PAYLOAD)} warmup={args.warmup} "
        f"samples={args.samples} rate={rate_hz:g} Hz (period={1000.0 * period_s:.3f} ms)"
    )

    # Warm-up: 建立连接、稳定 ZMQ 状态
    for _ in range(max(args.warmup, 0)):
        sock.send(PING_PAYLOAD)
        _ = sock.recv()
    if args.warmup > 0:
        print(f"[laptop_zmqtest] warm-up done ({args.warmup} pings)")

    rtts_ms: list[float] = []
    # Next scheduled send time (soft real-time: if RTT > period, resync after recv).
    next_send_t = time.perf_counter()
    try:
        for i in range(args.samples):
            now = time.perf_counter()
            if now < next_send_t:
                time.sleep(next_send_t - now)
            t0 = time.perf_counter()
            sock.send(PING_PAYLOAD)
            pong = sock.recv()
            t1 = time.perf_counter()
            if pong != PING_PAYLOAD:
                print(f"[warn] sample {i}: echo mismatch len={len(pong)}")
            rtt_ms = (t1 - t0) * 1000.0
            rtts_ms.append(rtt_ms)
            print(f"[{i + 1}/{args.samples}] RTT={rtt_ms:7.3f} ms")
            next_send_t += period_s
            if next_send_t < t1:
                next_send_t = t1 + period_s
    except zmq.Again:
        print("[error] recv/send timeout; check Pi script and firewall.")
        raise
    finally:
        sock.close(0)
        ctx.term()

    s = sorted(rtts_ms)
    mean_ms = statistics.fmean(rtts_ms)
    med_ms = statistics.median(rtts_ms)
    stdev_ms = statistics.stdev(rtts_ms) if len(rtts_ms) > 1 else 0.0
    p95_ms = percentile(s, 0.95)
    p99_ms = percentile(s, 0.99)
    min_ms = min(rtts_ms)
    max_ms = max(rtts_ms)

    print()
    print("--- RTT summary (ms) ---")
    print(f"  n={len(rtts_ms)}  mean={mean_ms:.3f}  std={stdev_ms:.3f}")
    print(f"  min={min_ms:.3f}  p50={med_ms:.3f}  p95={p95_ms:.3f}  p99={p99_ms:.3f}  max={max_ms:.3f}")
    print(f"  rough one-way ~ RTT/2: {0.5 * mean_ms:.3f} ms (symmetry assumption)")


if __name__ == "__main__":
    main()
