#!/usr/bin/env python3
"""
Compute max velocity/acceleration for one joint in trace_sim.csv.
Uses both position differentiation and direct velocity readings (with central difference)
to avoid overestimating acceleration from single-step RL control jerks.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple


def load_data(csv_path: Path, pos_col: str, vel_col: str) -> Tuple[List[float], List[float], List[float] | None]:
    t_sec: List[float] = []
    pos: List[float] = []
    vel: List[float] = []
    
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
            
        has_vel = vel_col in reader.fieldnames
        
        for row in reader:
            try:
                t = float(row["t_sec"])
                q = float(row[pos_col])
                v = float(row[vel_col]) if has_vel else 0.0
            except (TypeError, ValueError, KeyError):
                continue
            t_sec.append(t)
            pos.append(q)
            if has_vel:
                vel.append(v)
                
    if len(t_sec) < 3:
        raise ValueError("Not enough valid samples (need at least 3 rows).")
        
    return t_sec, pos, (vel if has_vel else None)


def central_diff(t: List[float], x: List[float]) -> Tuple[List[float], List[float]]:
    """
    Central difference: dx/dt = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
    This inherently smooths over a 2-step window, filtering out 1-step control noise.
    """
    d: List[float] = []
    t_out: List[float] = []
    for i in range(1, len(x) - 1):
        dt = t[i+1] - t[i-1]
        if dt <= 0.0:
            continue
        d.append((x[i+1] - x[i-1]) / dt)
        t_out.append(t[i])
    return d, t_out


def forward_diff(t: List[float], x: List[float]) -> Tuple[List[float], List[float]]:
    """Standard 1-step forward difference (sensitive to high-frequency noise)."""
    d: List[float] = []
    t_out: List[float] = []
    for i in range(1, len(x)):
        dt = t[i] - t[i-1]
        if dt <= 0.0:
            continue
        d.append((x[i] - x[i-1]) / dt)
        t_out.append(t[i])
    return d, t_out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_csv = repo_root / "policy" / "myrobot" / "trace_sim.csv"

    parser = argparse.ArgumentParser(description="Compute robust max |velocity| and |acceleration|.")
    parser.add_argument("--csv", type=Path, default=default_csv, help="Path to trace CSV.")
    parser.add_argument("--col", type=str, default="dof_pos_0", help="Position column name.")
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else (repo_root / args.csv)
    # Auto-infer velocity column name
    vel_col = args.col.replace("pos", "vel")
    
    t, q, v_log = load_data(csv_path, args.col, vel_col)
    
    print(f"=== Robust Acceleration Analysis ===")
    print(f"File   : {csv_path.name}")
    print(f"Joint  : {args.col}")
    print(f"Samples: {len(q)}\n")
    
    # 1. Theoretical / Envelope Estimation
    q_min, q_max = min(q), max(q)
    amp = (q_max - q_min) / 2.0
    print(f"[1. Position Envelope]")
    print(f"  Range: [{q_min:.4f}, {q_max:.4f}] rad")
    print(f"  Amplitude (A): {amp:.4f} rad\n")

    # 2. Velocity
    if v_log:
        v_max_log = max(v_log, key=abs)
        print(f"[2. Velocity (from {vel_col})]")
        print(f"  Max |v|: {abs(v_max_log):.4f} rad/s (signed: {v_max_log:.4f})")
    else:
        v_diff, _ = central_diff(t, q)
        v_max_log = max(v_diff, key=abs)
        print(f"[2. Velocity (central diff)]")
        print(f"  Max |v|: {abs(v_max_log):.4f} rad/s (signed: {v_max_log:.4f})")
        v_log = v_diff
        
    freq_est = abs(v_max_log) / (2 * math.pi * amp) if amp > 0 else 0
    print(f"  Estimated dominant frequency: {freq_est:.2f} Hz\n")
    
    # 3. Acceleration
    # Raw 1-step difference (susceptible to RL control jerks)
    a_raw, _ = forward_diff(t, v_log)
    a_raw_max = max(a_raw, key=abs) if a_raw else 0.0
    
    # Central difference (smooths over 2 timesteps)
    a_cen, _ = central_diff(t, v_log)
    a_cen_max = max(a_cen, key=abs) if a_cen else 0.0
    
    # Theoretical max acceleration for sine wave envelope
    a_theo = amp * (2 * math.pi * freq_est)**2
    
    print(f"[3. Acceleration]")
    print(f"  Theoretical |a| (Pure Sine) : {a_theo:.2f} rad/s^2 (Ideal physical envelope)")
    print(f"  Smoothed    |a| (Cent Diff) : {abs(a_cen_max):.2f} rad/s^2 (Filters out 1-step jumps)")
    print(f"  Raw 1-step  |a| (Fwd Diff)  : {abs(a_raw_max):.2f} rad/s^2 (Includes high-freq RL noise)")


if __name__ == "__main__":
    main()
