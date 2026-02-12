#!/usr/bin/env python3
"""DDoS detection in web server logs using regression analysis.

- Aggregates requests per minute
- Builds a robust regression baseline on log(counts)
- Detects anomalous intervals using residual z-scores
- Saves plots to task_3/

Usage:
    python task_3/ddos_detect.py
"""

import re
import os
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOG_PATH = "o_tagviashvili25_61845_server.log"

TS_RE = re.compile(
    r"\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+\d{2}:\d{2})\]"
)


def count_requests_per_minute(path: str):
    counts = defaultdict(int)
    min_dt, max_dt = None, None
    n = 0

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
            n += 1

    idx = pd.date_range(
        min_dt.replace(second=0, microsecond=0),
        max_dt.replace(second=0, microsecond=0),
        freq="min",
        tz=min_dt.tzinfo,
    )
    y = np.array([counts.get(ts.to_pydatetime(), 0) for ts in idx], dtype=float)
    return idx, y, min_dt, max_dt, n


def robust_log_linear_baseline(y: np.ndarray):
    t = np.arange(len(y), dtype=float)
    logy = np.log1p(y)

    mask = np.ones(len(y), dtype=bool)

    # Iteratively fit and drop the highest positive residuals (robust to spikes)
    for _ in range(10):
        slope, intercept = np.polyfit(t[mask], logy[mask], 1)
        pred = slope * t + intercept
        resid = logy - pred

        # Drop top 15% positive residuals from the fit set
        thr = np.quantile(resid[mask], 0.85)
        newmask = mask & (resid <= thr)
        if newmask.sum() == mask.sum():
            break
        mask = newmask

    pred = slope * t + intercept
    return pred, mask


def detect_attack_minutes(idx, y, pred_log, fit_mask):
    logy = np.log1p(y)
    resid = logy - pred_log

    # Robust scale estimate using MAD
    med = np.median(resid[fit_mask])
    mad = np.median(np.abs(resid[fit_mask] - med))
    sigma = 1.4826 * mad if mad > 0 else resid[fit_mask].std()
    z = resid / sigma

    core = z > 5
    expanded = core.copy()

    # Expand the core window to adjacent minutes with z>3
    for i in range(len(z)):
        if core[i]:
            j = i - 1
            while j >= 0 and z[j] > 3:
                expanded[j] = True
                j -= 1
            j = i + 1
            while j < len(z) and z[j] > 3:
                expanded[j] = True
                j += 1

    intervals = []
    i = 0
    while i < len(expanded):
        if not expanded[i]:
            i += 1
            continue
        start = idx[i]
        while i + 1 < len(expanded) and expanded[i + 1]:
            i += 1
        end = idx[i] + pd.Timedelta(minutes=1) - pd.Timedelta(seconds=1)
        intervals.append((start, end))
        i += 1

    return z, intervals


def save_plots(out_dir, idx, y, pred_log, expanded_mask, z):
    os.makedirs(out_dir, exist_ok=True)

    pred_min = np.expm1(pred_log)

    # Plot 1: requests per minute + baseline
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
    plt.xlabel("Time (UTC offset from log)")
    plt.ylabel("Requests per minute")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "requests_per_minute_regression.png"), dpi=200)
    plt.close()

    # Plot 2: z-scores over time
    plt.figure()
    plt.plot(idx.to_pydatetime(), z, label="Residual z-score (log-space)")
    plt.axhline(3, linestyle="--", label="z=3")
    plt.axhline(5, linestyle="--", label="z=5 (core)")
    plt.xlabel("Time (UTC offset from log)")
    plt.ylabel("z-score")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_zscore.png"), dpi=200)
    plt.close()


def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"Log file not found: {LOG_PATH}. Put it in repo root or update LOG_PATH."
        )

    idx, y, min_dt, max_dt, n = count_requests_per_minute(LOG_PATH)
    pred_log, fit_mask = robust_log_linear_baseline(y)
    z, intervals = detect_attack_minutes(idx, y, pred_log, fit_mask)

    expanded = np.zeros(len(z), dtype=bool)
    # rebuild expanded mask from intervals for plotting
    for start, end in intervals:
        # end is inclusive; convert to minute indices
        start_i = idx.get_loc(start)
        end_i = idx.get_loc(end.ceil("min") - pd.Timedelta(minutes=1))
        expanded[start_i : end_i + 1] = True

    out_dir = os.path.dirname(__file__)
    save_plots(out_dir, idx, y, pred_log, expanded, z)

    print(f"Log duration: {min_dt} -> {max_dt}")
    print(f"Total requests parsed: {n}")
    peak_i = int(np.argmax(y))
    print(f"Peak minute: {idx[peak_i]} with {int(y[peak_i])} requests/min")

    if not intervals:
        print("No DDoS intervals detected with current thresholds.")
    else:
        print("Detected DDoS interval(s):")
        for s, e in intervals:
            print(f" - {s} -> {e}")


if __name__ == "__main__":
    main()
