#!/usr/bin/env python3
"""Compare deployed policy_obs_sim.csv against Isaac ampobs.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

OBS_TERMS = [
    ("joint_pos", slice(0, 21)),
    ("joint_vel", slice(21, 42)),
    ("root_z", slice(42, 43)),
    ("ref_body_quat_tan_norm", slice(43, 49)),
    ("key_body_pos_relative", slice(49, 88)),
    ("progress", slice(88, 89)),
]


def load_amp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    obs_cols = [f"obs_{i}" for i in range(89)]
    act_cols = [f"action_{i}" for i in range(21)]
    df["progress"] = df["obs_88"].astype(float)
    df["obs_vec"] = df[obs_cols].astype(float).values.tolist()
    df["action_vec"] = df[act_cols].astype(float).values.tolist()
    return df


def _repair_merged_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Recover rows where progress was glued to obs_0 (missing comma in CSV logger)."""
    prog_re = re.compile(r"^([0-9]+\.[0-9]{6})(-?[0-9]+\.[0-9]+)$")

    def needs_repair(value: object) -> bool:
        text = str(value)
        try:
            float(text)
            return False
        except ValueError:
            return bool(prog_re.match(text))

    if not df["progress"].map(needs_repair).any():
        return df

    fixed = df.copy()
    prog_vals: list[float] = []
    obs0_vals: list[float] = []
    for row_idx, value in enumerate(fixed["progress"]):
        text = str(value)
        match = prog_re.match(text)
        if not match:
            prog_vals.append(float(text))
            obs0_vals.append(float(fixed.at[row_idx, "obs_0"]))
            continue
        prog_vals.append(float(match.group(1)))
        obs0_vals.append(float(match.group(2)))

    fixed["progress"] = prog_vals
    for i in range(88, 0, -1):
        fixed[f"obs_{i}"] = df[f"obs_{i - 1}"]
    fixed["obs_0"] = obs0_vals
    return fixed


def load_policy_obs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = _repair_merged_progress(df)
    obs_cols = sorted(
        [c for c in df.columns if c.startswith("obs_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )
    act_cols = sorted(
        [c for c in df.columns if c.startswith("action_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )
    for col in obs_cols + act_cols + ["progress", "episode_step"]:
        df[col] = df[col].astype(float)
    df["obs_vec"] = df[obs_cols].astype(float).values.tolist()
    df["action_vec"] = df[act_cols].astype(float).values.tolist()
    return df


def nearest_amp_row(amp: pd.DataFrame, progress: float) -> int:
    return int((amp["progress"] - progress).abs().idxmin())


def summarize_diff(deploy_obs: np.ndarray, amp_obs: np.ndarray) -> None:
    print(f"{'term':28s} {'mean|diff|':>12s} {'max|diff|':>12s}")
    print("-" * 56)
    for name, sl in OBS_TERMS:
        d = deploy_obs[sl] - amp_obs[sl]
        print(f"{name:28s} {np.mean(np.abs(d)):12.6f} {np.max(np.abs(d)):12.6f}")
    print("-" * 56)
    all_d = deploy_obs - amp_obs
    print(f"{'ALL (89)':28s} {np.mean(np.abs(all_d)):12.6f} {np.max(np.abs(all_d)):12.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-obs",
        type=Path,
        default=Path(__file__).with_name("policy_obs_sim.csv"),
    )
    parser.add_argument(
        "--ampobs",
        type=Path,
        default=Path(__file__).with_name("ampobs.csv"),
    )
    parser.add_argument("--steps", type=int, nargs="*", default=[1, 2, 3, 6, 10, 50],
                        help="episode_step values from policy_obs_sim.csv")
    args = parser.parse_args()

    if not args.policy_obs.exists():
        raise SystemExit(f"Missing {args.policy_obs}. Run MuJoCo dance first.")
    if not args.ampobs.exists():
        raise SystemExit(f"Missing {args.ampobs}.")

    amp = load_amp(args.ampobs)
    deploy = load_policy_obs(args.policy_obs)

    print(f"policy_obs rows: {len(deploy)}, ampobs rows: {len(amp)}")
    print()

    for step in args.steps:
        rows = deploy[deploy["episode_step"] == step]
        if rows.empty:
            print(f"[step {step}] not found in policy_obs_sim.csv")
            continue
        row = rows.iloc[0]
        deploy_obs = np.array(row["obs_vec"], dtype=float)
        deploy_act = np.array(row["action_vec"], dtype=float)
        # Match Isaac clip by progress term inside the policy observation vector.
        progress = float(deploy_obs[88])
        amp_idx = nearest_amp_row(amp, progress)
        amp_row = amp.iloc[amp_idx]
        amp_obs = np.array(amp_row["obs_vec"], dtype=float)
        amp_act = np.array(amp_row["action_vec"], dtype=float)

        print(f"=== deploy step={step}, progress={progress:.4f}  <->  amp row={amp_idx}, progress={amp_row['progress']:.4f} ===")
        summarize_diff(deploy_obs, amp_obs)
        act_diff = deploy_act - amp_act
        if np.isnan(deploy_act).any():
            print("action diff: skipped (legacy CSV row has NaN actions; re-run sim after rebuild)")
        else:
            print(f"action L2 diff: {np.linalg.norm(act_diff):.4f}, mean|action diff|: {np.mean(np.abs(act_diff)):.4f}")
        print()


if __name__ == "__main__":
    main()
