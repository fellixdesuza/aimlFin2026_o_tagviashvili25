# Task 3 — Detect DDoS time interval(s) using **regression analysis** (web server log)

## Goal
Given a web server event log, determine the **time interval(s)** where a **DDoS attack** happened.  
Requirement: use **regression analysis**, provide reproducible steps, code fragments, and visualizations.

---

## Dataset
- **Provided log file (place in repository root):** `o_tagviashvili25_61845_server.log`
- **Repository link (relative path expected by the script):**  
  [`o_tagviashvili25_61845_server.log`](o_tagviashvili25_61845_server.log)

### What the log looks like
Each event line contains a timestamp in square brackets, for example:

```text
[2024-03-22 18:40:12+04:00] ...
```

So we can extract time with a regex and parse it using `datetime.fromisoformat()`.

---

## Tools / environment
Tested with Python **3.10+**.

Install dependencies:

```bash
pip install pandas numpy matplotlib
```

---

## Method

### 1) Parse timestamps and build a requests-per-minute time series
1. Read the log line by line.
2. Extract the timestamp: `[YYYY-MM-DD HH:MM:SS+HH:MM]`
3. Convert it to a Python `datetime` object (timezone-aware).
4. Round down to the minute (`second=0`) and count events per minute.

Result: a time series:

- `minute_index` = every minute between the first and last log event
- `req_per_min` = how many log events occurred in that minute

For this dataset:
- Log duration: **2024-03-22 18:00:01 +0400 → 2024-03-22 19:00:59 +0400**
- Total parsed requests: **71,225**
- Minutes in the series: **61**

---

### 2) Regression baseline (normal traffic model)
A DDoS attack typically appears as an **abnormal spike** above normal request volume.  
To quantify “normal”, I built a **regression baseline** for request rate.

Because request counts are heavy-tailed (spiky and non-negative), I fit the regression in **log-space**:

- Target:  
  \[
  y = \log(1 + \text{req\_per\_min})
  \]
- Predictor:  
  \[
  x = \text{minute index}\ (0,1,2,\dots)
  \]
- Model:  
  \[
  \hat{y} = a x + b
  \]

#### Why “robust” regression?
If we fit a normal regression directly, large spikes can **pull the baseline upward** and “hide” the attack.  
To avoid this, I used an **iterative robust fitting** strategy:

1. Fit a regression.
2. Compute residuals `resid = y - y_hat`.
3. Remove the top **15%** positive residual points from the fit set.
4. Repeat a few iterations until stable.

This keeps the baseline focused on the *normal* traffic trend.

---

### 3) Residual analysis (how we decide what is an attack)
After fitting the baseline, compute residuals:

\[
r = y - \hat{y}
\]

Then convert residuals to a **z-score** so we can use a consistent threshold:

- First estimate scale robustly using **MAD** (Median Absolute Deviation):
  \[
  \sigma \approx 1.4826 \cdot \text{MAD}(r)
  \]
- Then:
  \[
  z = \frac{r}{\sigma}
  \]

Interpretation:
- `z` measures how many “robust standard deviations” above baseline a minute is.
- Large positive `z` = unusually high traffic → likely attack.

---

### 4) Turning anomalous minutes into **attack interval(s)**
To get a clean time interval (not just single minutes), I used a **two-level rule**:

1. **Core attack minutes:** `z > 5`  
   (very strong anomalies)
2. **Expanded window:** include neighbor minutes where `z > 3`  
   (captures ramp-up / ramp-down around the core)

Finally, merge consecutive expanded minutes into continuous interval(s).

This gives an interval with:
- clear statistical justification (core)
- realistic boundaries (expanded)

---

## Result — DDoS time interval(s)

### Final detected interval (timezone from the log: **UTC+04:00**)
- **Start:** **2024-03-22 18:37:00 +0400**
- **End:** **2024-03-22 18:44:59 +0400**
- Duration: **8 minutes** (inclusive minutes: 18:37 … 18:44)

### Why this interval is classified as DDoS (detailed)
Inside this window:

- Total requests in interval: **33,182**
- Share of all requests: **46.59%** of the entire log traffic happens in only **8 minutes**
- Average rate during interval: **≈ 4,148 req/min**
- Peak minute: **2024-03-22 18:40:00 +0400** with **10,773 req/min**

This is consistent with DDoS behavior: **very high, sustained request volume** over a short period, far above the regression baseline.

### Minute-by-minute evidence (observed vs baseline vs z-score)
The table below shows the exact minutes included in the expanded interval and how they compare to the regression baseline:

| Minute (start) | Req/min | Baseline (req/min) | z-score | Core? (z>5) |
|---|---:|---:|---:|:---:|
| 2024-03-22 18:37:00 +0400 | 1213 | 346.8 | 4.68 | no |
| 2024-03-22 18:38:00 +0400 | 1412 | 345.0 | 5.26 | YES |
| 2024-03-22 18:39:00 +0400 | 10744 | 343.1 | 12.87 | YES |
| 2024-03-22 18:40:00 +0400 | 10773 | 341.4 | 12.90 | YES |
| 2024-03-22 18:41:00 +0400 | 849 | 339.6 | 3.42 | no |
| 2024-03-22 18:42:00 +0400 | 3508 | 337.8 | 8.74 | YES |
| 2024-03-22 18:43:00 +0400 | 3720 | 336.0 | 8.98 | YES |
| 2024-03-22 18:44:00 +0400 | 963 | 334.3 | 3.95 | no |


Notes:
- Minutes with **Core? = YES** are the strongest anomalies (`z > 5`).
- Minutes with `z` between 3 and 5 (e.g., 18:37, 18:41, 18:44) are included because they sit next to core minutes and represent the **attack ramp-up / ramp-down**.

### Did we find multiple attack windows?
With the chosen thresholds (`core z>5`, expansion `z>3`), the script detected **one** contiguous DDoS interval in this log:
- **2024-03-22 18:37:00 +0400 → 2024-03-22 18:44:59 +0400**

If you lower thresholds, you may detect additional “minor spikes”, but they will be less statistically extreme.

---

## Visualizations

### 1) Requests per minute + regression baseline
![](requests_per_minute_regression.png)

How to read:
- Line “Observed req/min” = real traffic per minute
- Line “Regression baseline” = expected normal traffic from regression
- Shaded regions = detected DDoS minutes (expanded interval)

### 2) Residual z-score over time
![](residual_zscore.png)

How to read:
- `z = 3` = expansion threshold
- `z = 5` = core threshold
- Attack minutes are where the curve stays above these thresholds

---

## Reproducibility

### Repository structure expected
```
repo-root/
  o_tagviashvili25_61845_server.log
  
    ddos_detect.py
    ddos.md
```

### Run the analysis
From **repo root**:

```bash
python ddos_detect.py
```

Expected output (same interval):
- **2024-03-22 18:37:00 +0400 → 2024-03-22 18:44:59 +0400**

### Output files produced
After running, you should have:

- `requests_per_minute_regression.png`
- `residual_zscore.png`
- `per_minute_regression.csv`

The CSV is important because it makes the analysis auditable. It contains:
- observed req/min
- regression baseline
- residual
- z-score
- flags (`is_core`, `is_expanded`)

---

## Main code fragments (core logic)

### A) Parse timestamps and count requests/minute
```python
TS_RE = re.compile(
    r"\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+\d{2}:\d{2})\]"
)

t = dt.datetime.fromisoformat(m.group(1))
tmin = t.replace(second=0, microsecond=0)
counts[tmin] += 1
```

### B) Robust regression baseline in log-space
```python
logy = np.log1p(y)  # y = req/min
slope, intercept = np.polyfit(t[mask], logy[mask], 1)
pred = slope * t + intercept

# remove top 15% positive residuals to keep baseline "normal"
resid = logy - pred
thr = np.quantile(resid[mask], 0.85)
mask = mask & (resid <= thr)
```

### C) z-score and interval extraction
```python
# robust scale via MAD
med = np.median(resid[fit_mask])
mad = np.median(np.abs(resid[fit_mask] - med))
sigma = 1.4826 * mad
z = resid / sigma

core = z > 5
expanded = expand_core_to_neighbors(core, z, z_expand=3)

intervals = merge_consecutive_minutes(expanded)
```

---

## Parameters you can tune (sensitivity analysis)
In `ddos_detect.py` you can change thresholds:

- `--z-core` (default 5.0): higher → fewer, stronger detections
- `--z-expand` (default 3.0): higher → tighter interval boundaries

Example:

```bash
python ddos_detect.py --z-core 6 --z-expand 4
```

---

## Conclusion
Using **robust regression analysis** on the log-transformed requests-per-minute series and residual **z-score** detection, the DDoS attack was identified as:

- **2024-03-22 18:37:00 +0400 → 2024-03-22 18:44:59 +0400 (UTC+04:00)**

This interval contains an extreme traffic surge (peak **10,773 req/min**) and accounts for **~46.59%** of all requests in the dataset, which is consistent with a DDoS event.
