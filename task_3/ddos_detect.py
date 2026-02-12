#!/usr/bin/env python3
"""Task 3 — Detect DDoS time interval(s) from a web server log using regression analysis.

What this script does
---------------------
1) Parses timestamps from each log line (expects: [YYYY-MM-DD HH:MM:SS+HH:MM])
2) Aggregates requests per minute
3) Fits a *robust* linear regression baseline in log-space:
      y = log(1 + req_per_min)
      x = minute_index
   Robustness: iteratively excludes the top 15% positive residuals so that spikes
   (potential attacks) do not pull the baseline upward.
4) Computes residual z-scores using a robust scale estimate (MAD).
5) Detects DDoS minutes and merges them into interval(s):
      - core minutes: z > 5
      - expanded window: extend core to neighbors with z > 3
6) Saves reproducible outputs in :
      - requests_per_minute_regression.png
      - residual_zscore.png
      - per_minute_regression.csv   (minute-level table: observed, baseline, residual, z, flags)

Usage
-----
From repo root (recommended):
    python ddos_detect.py

If your log file is elsewhere:
    python ddos_detect.py --log /path/to/server.log

Dependencies:
    pip install pandas numpy matplotlib
"""

import argparse
import os
import re
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TS_RE = re.compile(
    r"\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+\d{2}:\d{2})\]"
)


def count_requests_per_minute(path: str):
    """Return a timezone-aware minute index and requests-per-minute array."""
    counts = defaultdict(int)
    min_dt, max_dt = None, None
    parsed = 0

    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = TS_RE.search(line)
            if not m:
                continue
            t = dt.datetime.fromisoformat(m.group(1))
            tmin = t.replace(second=0, microsecond=0)
            counts[tmin] += 1

            min_dt = t if (min_dt is None or t < min_dt) else min_dt
            max_dt = t if (max_dt is None or t > max_dt) else max_dt
            parsed += 1

    if parsed == 0:
        raise ValueError("No timestamps were parsed. Check the log format / regex.")

    idx = pd.date_range(
        min_dt.replace(second=0, microsecond=0),
        max_dt.replace(second=0, microsecond=0),
        freq="min",
        tz=min_dt.tzinfo,
    )
    y = np.array([counts.get(ts.to_pydatetime(), 0) for ts in idx], dtype=float)
    return idx, y, min_dt, max_dt, parsed


def robust_log_linear_baseline(y: np.ndarray, iterations: int = 10, drop_q: float = 0.85):
    """Fit y_log = a*t + b robustly by excluding high positive residuals."""
    t = np.arange(len(y), dtype=float)
    logy = np.log1p(y)

    mask = np.ones(len(y), dtype=bool)
    slope = intercept = None

    for _ in range(iterations):
        slope, intercept = np.polyfit(t[mask], logy[mask], 1)
        pred = slope * t + intercept
        resid = logy - pred

        thr = np.quantile(resid[mask], drop_q)  # drop top (1-drop_q) of positive residuals
        newmask = mask & (resid <= thr)
        if newmask.sum() == mask.sum():
            break
        mask = newmask

    pred = slope * t + intercept
    return pred, mask


def detect_attack(idx, y, pred_log, fit_mask, z_core: float = 5.0, z_expand: float = 3.0):
    """Return z-scores and merged attack intervals."""
    logy = np.log1p(y)
    resid = logy - pred_log

    # Robust z-score using MAD (Median Absolute Deviation)
    med = np.median(resid[fit_mask])
    mad = np.median(np.abs(resid[fit_mask] - med))
    sigma = 1.4826 * mad if mad > 0 else resid[fit_mask].std(ddof=1)
    z = resid / sigma

    core = z > z_core
    expanded = core.copy()

    # Expand each core region to adjacent minutes with moderately high z-score
    for i in range(len(z)):
        if core[i]:
            j = i - 1
            while j >= 0 and z[j] > z_expand:
                expanded[j] = True
                j -= 1
            j = i + 1
            while j < len(z) and z[j] > z_expand:
                expanded[j] = True
                j += 1

    # Merge expanded minutes into continuous intervals
    intervals = []
    i = 0
    while i < len(expanded):
        if not expanded[i]:
            i += 1
            continue
        start = idx[i]
        while i + 1 < len(expanded) and expanded[i + 1]:
            i += 1
        # inclusive end: end of that minute (mm:59)
        end = idx[i] + pd.Timedelta(minutes=1) - pd.Timedelta(seconds=1)
        intervals.append((start, end))
        i += 1

    return z, resid, core, expanded, intervals, sigma


def save_plots(out_dir, idx, y, pred_log, expanded_mask, z, z_core=5.0, z_expand=3.0):
    os.makedirs(out_dir, exist_ok=True)
    pred_min = np.expm1(pred_log)

    # Plot 1: requests/min vs baseline
    plt.figure()
    plt.plot(idx.to_pydatetime(), y, label="Observed req/min")
    plt.plot(idx.to_pydatetime(), pred_min, label="Regression baseline (exp of log-fit)")
    for i in range(len(idx)):
        if expanded_mask[i]:
            plt.axvspan(
                idx[i].to_pydatetime(),
                (idx[i] + pd.Timedelta(minutes=1)).to_pydatetime(),
                alpha=0.2,
            )
    plt.xlabel("Time (timezone from log)")
    plt.ylabel("Requests per minute")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "requests_per_minute_regression.png"), dpi=200)
    plt.close()

    # Plot 2: residual z-scores
    plt.figure()
    plt.plot(idx.to_pydatetime(), z, label="Residual z-score (log-space)")
    plt.axhline(z_expand, linestyle="--", label=f"z={z_expand} (expand)")
    plt.axhline(z_core, linestyle="--", label=f"z={z_core} (core)")
    plt.xlabel("Time (timezone from log)")
    plt.ylabel("z-score")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_zscore.png"), dpi=200)
    plt.close()


def interval_stats(df, interval):
    s, e = interval
    # df['minute'] is minute start; interval end is inclusive
    mask = (df["minute"] >= s) & (df["minute"] <= e.floor("min"))
    part = df.loc[mask].copy()
    return {
        "start": s,
        "end": e,
        "minutes": int(mask.sum()),
        "total_requests": int(part["req_per_min"].sum()),
        "avg_req_per_min": float(part["req_per_min"].mean()),
        "peak_req_per_min": int(part["req_per_min"].max()),
        "peak_minute": part.loc[part["req_per_min"].idxmax(), "minute"],
        "max_z": float(part["z"].max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default=os.path.join(os.path.dirname(__file__), "..", "o_tagviashvili25_61845_server.log"),
        help="Path to the server log file",
    )
    parser.add_argument("--z-core", type=float, default=5.0, help="Core anomaly threshold")
    parser.add_argument("--z-expand", type=float, default=3.0, help="Expansion threshold")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"Log file not found: {args.log}")

    idx, y, min_dt, max_dt, parsed = count_requests_per_minute(args.log)
    pred_log, fit_mask = robust_log_linear_baseline(y)
    z, resid, core, expanded, intervals, sigma = detect_attack(
        idx, y, pred_log, fit_mask, z_core=args.z_core, z_expand=args.z_expand
    )

    # Build a minute-level reproducible table
    df = pd.DataFrame(
        {
            "minute": idx,
            "req_per_min": y.astype(int),
            "baseline_req_per_min": np.expm1(pred_log),
            "resid_log": resid,
            "z": z,
            "is_core": core,
            "is_expanded": expanded,
        }
    )
    df["baseline_req_per_min"] = df["baseline_req_per_min"].round(3)
    df["resid_log"] = df["resid_log"].round(6)
    df["z"] = df["z"].round(6)

    out_dir = os.path.dirname(__file__)
    df.to_csv(os.path.join(out_dir, "per_minute_regression.csv"), index=False)

    save_plots(out_dir, idx, y, pred_log, expanded, z, z_core=args.z_core, z_expand=args.z_expand)

    # Console summary (useful for the report)
    print(f"Log duration: {min_dt} -> {max_dt}")
    print(f"Total parsed requests: {parsed}")
    print(f"Minutes in series: {len(df)}")
    print(f"Robust scale (sigma) in log-space: {sigma:.6f}")
    peak_i = int(np.argmax(y))
    print(f"Global peak minute: {idx[peak_i]} with {int(y[peak_i])} req/min")

    total_all = int(df["req_per_min"].sum())

    if not intervals:
        print("No DDoS intervals detected with current thresholds.")
        return

    print("\nDetected DDoS interval(s):")
    for (s, e) in intervals:
        stats = interval_stats(df, (s, e))
        pct = 100.0 * stats["total_requests"] / total_all if total_all else 0.0
        print(
            f" - {stats['start']} -> {stats['end']} | "
            f"{stats['minutes']} min | total={stats['total_requests']} ({pct:.2f}%) | "
            f"avg={stats['avg_req_per_min']:.1f}/min | peak={stats['peak_req_per_min']} at {stats['peak_minute']} | "
            f"max z={stats['max_z']:.2f}"
        )


if __name__ == "__main__":
    main()
