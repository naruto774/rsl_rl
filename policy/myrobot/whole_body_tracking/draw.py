#!/usr/bin/env python3
"""Plot selected CSV columns (by index) on one figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============ config ============
FILE = "../trace.csv"
COL_START = 75
COL_END = 76  # inclusive
TIME_COL = 88  # x-axis column index; set None to use row index / POLICY_HZ
POLICY_HZ = 60.0
EPISODE_DURATION_S = 21.7
SAVE_PATH = None  # e.g. "ampobs_obs0_21.png"
# ================================


def main() -> None:
    path = (Path(__file__).parent / FILE).resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    df = pd.read_csv(path)
    ncols = df.shape[1]
    if COL_START < 0 or COL_END >= ncols or COL_START > COL_END:
        raise ValueError(f"Column range [{COL_START}, {COL_END}] out of bounds (ncols={ncols})")

    if TIME_COL is not None:
        if TIME_COL < 0 or TIME_COL >= ncols:
            raise ValueError(f"TIME_COL={TIME_COL} out of bounds (ncols={ncols})")
        x = df.iloc[:, TIME_COL].astype(float) * EPISODE_DURATION_S
        x_label = "time (s)"
    else:
        x = df.index / POLICY_HZ
        x_label = "time (s)"

    fig, ax = plt.subplots(figsize=(14, 7))
    for col_idx in range(COL_START, COL_END + 1):
        ax.plot(x, df.iloc[:, col_idx].astype(float), lw=1.0, label=str(col_idx))

    ax.set_title(f"{FILE}: col {COL_START} .. {COL_END}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        ncol=1,
        frameon=False,
    )

    fig.tight_layout()
    if SAVE_PATH is not None:
        out = Path(__file__).with_name(SAVE_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
