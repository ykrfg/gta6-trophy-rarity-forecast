"""Trophy rarity forecast for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of GTA VI on PS5, five quantiles (5/25/50/75/95%)
of its earn rate on Exophase at the case's target time, from the recorded
history of the trophy page.

Model (own design, standard library only, deterministic, offline):

1. Collapse the history to distinct Exophase states: consecutive reads with
   identical (players_tracked, rates) are one refresh state. All fitting
   happens on distinct states.
2. Median: fit logit(rate) against a saturating basis u(t) = 1 - exp(-t/T),
   t = hours since release, with T chosen per case by grid search minimising
   the pooled fit error across trophies. A saturating basis is essential:
   earn rates level off as players progress, and a log-linear fit of
   earners (or of logit rates) extrapolates the steep early growth far past
   the data -- on a 4.5-day horizon that blows rates up toward 100%.
   Slopes are shrunk toward the tier-pooled slope. The fitted rate path
   already carries the dilution effect (new, less-progressed players push
   rates down at refreshes), because it is fitted to the observed rates,
   which are earners / players_tracked at each refresh.
3. Spread per trophy from three pooled components, combined in quadrature:
   (a) the per-trophy fit residual, shrunk toward the pooled residual;
   (b) the empirical refresh-jump scale: the RMS logit jump between
   consecutive distinct states in this very history, scaled by the square
   root of the expected number of refreshes over the horizon (with a prior
   jump scale / refresh cadence when the history shows no refresh yet);
   (c) a small random-walk diffusion growing with sqrt(days), plus a term
   for trend uncertainty over the extrapolated part of the basis.
   Quantiles are inv_logit(median_logit + z_tau * s_i).
4. Flat / short history falls back gracefully to persistence with a wide,
   refresh-aware spread. Nulls, a single read, and 0% / 100% rates are all
   handled (rates are clamped to [0.01, 99.99] before any logit, as the
   scorer does).

Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Prior refresh behaviour, used only while the history shows no refresh yet:
# a refresh on launch week can move a slow trophy's rate several-fold
# (the published launch-night refresh took the tracked population 638 ->
# 3,868 while most trophies gained almost no earners, so their logit rates
# fell by O(1)), and refreshes arrive hours apart. These are priors, not
# fitted values.
PRIOR_JUMP_LOGIT = 1.25
PRIOR_REFRESH_GAP_MIN = 240.0
# Diffusion of the logit rate per sqrt(day), beyond refresh jumps.
DIFFUSION_PER_SQRT_DAY = 0.10
# Shrinkage strength (pseudo-count) for per-trophy slopes / residuals.
SHRINK_K = 3.0
SHRINK_M = 4.0


def parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime (handles 'Z' and fractions)."""
    if text is None:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x)) if x < 700 else 100.0
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    # Coefficients for the central region and the tails.
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, rms_residual) or None."""
    n = len(xs)
    if n == 0:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return (my, 0.0, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        rms = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        rms = 0.0
    return (a, b, rms)


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _same_state(a, b):
    """True when two reads show the same Exophase page: same players count
    and no trophy whose rate is known in both reads disagrees. A read that
    merely lacks some trophies (nulls) does not by itself start a new state."""
    if a.get("players_tracked") != b.get("players_tracked"):
        return False
    ra = a.get("rates") or []
    rb = b.get("rates") or []
    for x, y in zip(ra, rb):
        if x is not None and y is not None and round(float(x), 6) != round(float(y), 6):
            return False
    return True


def distinct_states(history):
    """Collapse consecutive reads of the same page into one refresh state."""
    states = []
    for read in history or []:
        if states and _same_state(states[-1], read):
            continue
        states.append(read)
    return states


def forecast_case(case):
    trophies = case.get("trophies") or []
    n_tr = len(trophies)
    levels = case.get("levels") or LEVELS
    zvals = [norm_ppf(level) for level in levels]
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = max(horizon_min, 0.0) / 1440.0

    game = case.get("game") or {}
    release = parse_time(game.get("release_date"))
    if release is None:
        release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case.get("target"))
    checkpoint = parse_time(case.get("at"))

    states = distinct_states(case.get("history") or [])
    if not states:
        # No history at all: a flat, very wide forecast is the honest answer.
        return {"trophies": [
            {"id": t.get("id"), "q": [round4(inv_logit_pct(logit(1.0) + z * 1.5)) for z in zvals]}
            for t in trophies]}

    def hours_since_release(dt):
        if dt is None:
            return 0.0
        return max((dt - release).total_seconds() / 3600.0, 0.0)

    # Hours since release for each distinct state, and for the target.
    ts = [hours_since_release(parse_time(st.get("at"))) for st in states]
    t_target = hours_since_release(target) if target is not None else (ts[-1] if ts else 0.0)

    # --- Refresh cadence and jump scale from the distinct states. ---
    state_times = [parse_time(st.get("at")) for st in states]
    gaps = []
    for i in range(1, len(state_times)):
        if state_times[i] is not None and state_times[i - 1] is not None:
            gaps.append((state_times[i] - state_times[i - 1]).total_seconds() / 60.0)
    gap_med = median_of(gaps) if gaps else PRIOR_REFRESH_GAP_MIN
    gap_med = max(gap_med, 30.0)
    expected_refreshes = horizon_min / gap_med if gap_med > 0 else 0.0

    # Logit jumps between consecutive distinct states, pooled over trophies.
    jumps = []
    for k in range(1, len(states)):
        r_prev = states[k - 1].get("rates") or []
        r_cur = states[k].get("rates") or []
        for i in range(min(n_tr, len(r_prev), len(r_cur))):
            if r_prev[i] is None or r_cur[i] is None:
                continue
            jumps.append(logit(float(r_cur[i])) - logit(float(r_prev[i])))
    if jumps:
        jump_rms = math.sqrt(sum(j * j for j in jumps) / len(jumps))
    else:
        jump_rms = PRIOR_JUMP_LOGIT

    # --- Last non-null rate per trophy (persistence fallback). ---
    last_rate = []
    for i in range(n_tr):
        rate = None
        for st in reversed(states):
            rates = st.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rate.append(rate if rate is not None else 1.0)

    # --- Saturating-basis fits of logit(rate). ---
    # Per-trophy points: (hours, logit(rate)) over distinct states.
    points = []
    for i in range(n_tr):
        pts = []
        for k, st in enumerate(states):
            rates = st.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                pts.append((ts[k], logit(float(rates[i]))))
        points.append(pts)

    # Choose the saturation timescale T per case: grid-search the T that
    # minimises the pooled squared residual of logit(rate) ~ a + b*u(t),
    # u(t) = 1 - exp(-t/T), over trophies with >= 3 points.
    T_GRID = [4.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0,
              144.0, 192.0, 288.0, 432.0]

    def sse_for_T(T):
        total = 0.0
        count = 0
        for pts in points:
            if len(pts) < 3:
                continue
            us = [1.0 - math.exp(-t / T) for t, _ in pts]
            ys = [y for _, y in pts]
            fit = ols(us, ys)
            if fit is None:
                continue
            a, b, _ = fit
            total += sum((y - (a + b * u)) ** 2 for u, y in zip(us, ys))
            count += len(pts)
        return (total / count) if count else None

    best_T = None
    best_sse = None
    if len(states) >= 3:
        for T in T_GRID:
            sse = sse_for_T(T)
            if sse is not None and (best_sse is None or sse < best_sse - 1e-12):
                best_sse = sse
                best_T = T

    fits = []
    u_target = None
    if best_T is not None:
        u_target = 1.0 - math.exp(-t_target / best_T)
        for pts in points:
            if len(pts) < 2:
                fits.append(None)
                continue
            us = [1.0 - math.exp(-t / best_T) for t, _ in pts]
            ys = [y for _, y in pts]
            fits.append(ols(us, ys))
    else:
        fits = [None] * n_tr

    # Tier-pooled slopes for shrinkage.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [f[1] for f, t in zip(fits, trophies)
                if f is not None and (t.get("tier") or "") == tier]
        tier_slopes[tier] = sum(vals) / len(vals) if vals else None
    all_slopes = [f[1] for f in fits if f is not None]
    global_slope = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0

    residuals = [f[2] for f in fits if f is not None and f[2] > 0]
    pooled_resid = median_of(residuals) if residuals else 0.0

    # Shrunk slopes, computed once (used for the median and for detrending).
    b_shrunk_list = []
    for i, trophy in enumerate(trophies):
        fit = fits[i]
        tier = trophy.get("tier") or ""
        pool_slope = tier_slopes.get(tier)
        if pool_slope is None:
            pool_slope = global_slope
        if fit is not None:
            n_pts = len(points[i])
            w = n_pts / (n_pts + SHRINK_K)
            b_shrunk_list.append(w * fit[1] + (1.0 - w) * pool_slope)
        else:
            b_shrunk_list.append(pool_slope)

    # Detrend the refresh jumps: subtract the fitted trend's move between
    # the two states, so the jump scale measures genuine refresh surprise
    # rather than the trend itself. (With no fit, the raw jumps stand.)
    if best_T is not None:
        u_states = [1.0 - math.exp(-t / best_T) for t in ts]
        djumps = []
        for k in range(1, len(states)):
            r_prev = states[k - 1].get("rates") or []
            r_cur = states[k].get("rates") or []
            du = u_states[k] - u_states[k - 1]
            for i in range(min(n_tr, len(r_prev), len(r_cur))):
                if r_prev[i] is None or r_cur[i] is None:
                    continue
                raw = logit(float(r_cur[i])) - logit(float(r_prev[i]))
                djumps.append(raw - b_shrunk_list[i] * du)
        if djumps:
            jump_rms = math.sqrt(sum(j * j for j in djumps) / len(djumps))

    # Refresh surprises accumulate only weakly: the answer is the state at
    # the last refresh before the target, so the horizon contributes a
    # capped number of effective refresh steps.
    s_jump = jump_rms * math.sqrt(min(expected_refreshes, 6.0))
    s_rw = DIFFUSION_PER_SQRT_DAY * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        fit = fits[i]

        if fit is not None and len(points[i]) >= 2 and u_target is not None:
            a, b, rms = fit
            b_shrunk = b_shrunk_list[i]
            # Anchor at the last observed rate and add only the fitted
            # move: the fitted level averages the saturated part of the
            # basis and would sit below the latest read when T is small.
            u_last = 1.0 - math.exp(-points[i][-1][0] / best_T)
            move = b_shrunk * (u_target - u_last)
            move = max(-2.5, min(2.5, move))
            med_logit = logit(last_rate[i]) + move
            median_rate = clamp_rate(inv_logit_pct(med_logit))
            # Residual scale, shrunk toward the pooled residual.
            s_fit = math.sqrt((SHRINK_M * pooled_resid ** 2 + rms * rms)
                              / (SHRINK_M + 1.0)) if rms > 0 else pooled_resid
            # Trend uncertainty over the extrapolated part of the basis.
            u_last = 1.0 - math.exp(-ts[-1] / best_T) if ts else 0.0
            s_extrap = 0.5 * abs(b_shrunk) * max(0.0, u_target - u_last)
        else:
            # Flat / single-state history: persistence median.
            median_rate = last_rate[i]
            s_fit = pooled_resid
            s_extrap = 0.0

        median_rate = clamp_rate(median_rate)
        med_logit = logit(median_rate)
        s_total = math.sqrt(s_fit * s_fit + s_jump * s_jump + s_rw * s_rw
                            + s_extrap * s_extrap)
        s_total = min(max(s_total, 0.08), 2.0)

        qs = []
        prev = -1.0
        for z in zvals:
            q = round4(inv_logit_pct(med_logit + z * s_total))
            q = min(100.0, max(0.0, q))
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy.get("id"), "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
