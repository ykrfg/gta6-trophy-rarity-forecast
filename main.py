"""Trophy rarity forecast for make.fun gta6-trophy-rarity-forecast-01.

For every GTA VI PS5 trophy, forecast the 5/25/50/75/95 % quantiles of its
Exophase earn rate at the case's target time, from the recorded page history.

Model (standard library only, deterministic, offline):

1. Collapse the history to distinct Exophase states: consecutive reads with
   identical (players_tracked, rates) are one refresh state.
2. Decompose the rate into a player base P(t) and per-trophy earners E_i(t).
   log P and y_i = log E_i - beta_i log P are fitted with saturating curves
   u(t) = 1 - exp(-t/T), model-averaged over a T grid (AIC weights plus a
   small uniform floor). The median rate is exp(y - (1-beta) log P), so
   both earner completion and dilution by newly tracked players are explicit.
3. beta_i is estimated from consecutive refresh increments, shrunk to the
   tier pool and then toward 1 (rates already carry most observed dilution).
   Flat/stale histories use beta = 0, i.e. the observed earners hold while the
   denominator grows -- the practice dilution pattern.
4. Spread combines fit/trend uncertainty, a beta-uncertainty term, the recent
   detrended refresh-jump RMS scaled by sqrt(expected refreshes), quantisation
   noise, diffusion, and a player-trend term. Its scale was calibrated on
   synthetic staircase backtests. A flat history widens the lower tail,
   because the next refresh can reveal a much larger tracked base.
5. Rare-refresh guard: when there are few distinct states or refreshes are
   hours apart, curve T is capped at the observed history span. A long-T
   curve fitted through launch day otherwise projects implausible growth over
   a multi-day horizon.
6. Flat / short history falls back to a drift-plus-sigma player prior with
   refresh-aware spread. Nulls, one read, 0 % and 100 % rates are handled;
   100 % trophies stay at 100 % with a narrow band.

Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

DEFAULT_CFG = {
    # curve grids (hours)
    "p_T_grid": [4.0, 8.0, 16.0, 24.0, 48.0, 96.0, 192.0, 384.0,
                 768.0, 1536.0, 3072.0],
    "e_T_grid": [4.0, 8.0, 16.0, 24.0, 48.0, 96.0, 192.0, 384.0,
                 768.0, 1536.0, 3072.0],
    "model_uniform": 0.03,       # uniform mixture over T models
    # Rare-refresh guard: with only a handful of distinct page states, a
    # long-T curve fitted through the first day can imply absurd growth over
    # a multi-day horizon.  Cap T at the observed history span in that case.
    "t_span_mult": 1.0,          # cap T at span*mult (hours)
    "t_span_min": 24.0,          # lower bound for the span-based T cap
    "t_cap_states": 12,          # apply cap when distinct states <= this
    "t_cap_gap_min": 90.0,       # apply cap when median refresh gap >= this
    "t_cap_horizon_min": 24.0,   # never cap short-horizon cases
    "t_cap_log_growth": 3.8,     # only cap when tracked base grew >= e^3.8
    "between_scale": 0.80,       # temper model-shape variance
    "p_half_life": 0.0,          # recency weighting half-life (hours), 0=off
    "e_half_life": 0.0,
    "short_h0": 12.0,            # below this horizon model-shape variance is damped
    "short_floor": 1.0,          # 1.0 disables the damping
    "short_pow": 0.5,
    "sigma_scale": 0.90,
    # player model
    "p_horizon_sigma": 0.10,     # extra log-P sd per sqrt(day)
    "p_sigma_floor": 0.05,
    "p_sigma_cap": 1.10,
    "flat_p_drift": 0.28,        # expected log growth per sqrt(day), flat
    "flat_p_sigma0": 0.22,
    "flat_p_sigma": 0.55,
    "flat_p_cap": 1.70,
    # earner model
    "shrink_k": 3.0,
    "e_use_min": 3,              # points needed to fit an earner curve
    "e_min_count": 1.0,          # ignore fractional / zero earner counts
    "e_reliable": 6.0,           # E_last for slope pooling
    "e_quant_sigma": 0.45,       # times 1/sqrt(E)
    "e_horizon_sigma": 0.05,
    "fallback_elasticity": 0.0,  # E_T = E_last * (P_T/P_last)**elast
    "e_anchor_obs": 1.00,        # weight on observed last log-earner level
    # earner / player coupling: log E = y + beta * log P
    "beta_prior": 0.0,           # used only when no refresh increments exist
    "beta_shrink_k": 5.0,
    "beta_min": -0.2,
    "beta_max": 1.3,
    "beta_sigma_floor": 0.08,
    "beta_sigma_cap": 0.50,
    "beta_no_data_sigma": 0.35,
    "beta_term_cap": 1.5,
    "beta_to_one": 0.9,          # shrink estimated beta toward 1
    "beta_override": None,       # debug/AB: force a constant beta
    # refresh jump / diffusion
    "jump_prior": 0.85,
    "jump_cap": 6.0,
    "jump_mult": 1.0,
    "jump_window": 40,           # recent distinct states used for jump scale
    "jump_shrink_k": 4.0,
    "diffusion": 0.08,
    "asym_lo": 0.4,
    "asym_hi": 1.0,
    # Flat history: a refresh can reveal a much larger tracked base, so the
    # rate tail is genuinely left-skewed (practice dilution pattern).
    "asym_flat_lo": 1.6,
    "asym_flat_hi": 0.7,
    # High-growth sparse history: less extreme lower tail than the flat case.
    "asym_cap_lo": 0.5,
    "asym_cap_hi": 1.0,
    # output
    "sigma_min": 0.08,
    "sigma_max": 2.2,
    "locked_sigma": 0.06,
    "locked_eps": 99.995,
}


def _cfg(cfg):
    out = dict(DEFAULT_CFG)
    if cfg:
        out.update(cfg)
    return out


def parse_time(text):
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
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p):
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


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _same_state(a, b):
    if a.get("players_tracked") != b.get("players_tracked"):
        return False
    ra = a.get("rates") or []
    rb = b.get("rates") or []
    for x, y in zip(ra, rb):
        if x is not None and y is not None and round(float(x), 6) != round(float(y), 6):
            return False
    return True


def distinct_states(history):
    states = []
    for read in history or []:
        if states and _same_state(states[-1], read):
            continue
        states.append(read)
    return states


# ---------------------------------------------------------------------------
# Saturating-curve helpers
# ---------------------------------------------------------------------------

def _fit_one(xs, ys, T, weights=None):
    us = [1.0 - math.exp(-x / T) for x in xs]
    n = len(xs)
    if weights is None:
        weights = [1.0] * n
    sw = sum(weights)
    if sw <= 0:
        return None
    mx = sum(w * u for w, u in zip(weights, us)) / sw
    my = sum(w * y for w, y in zip(weights, ys)) / sw
    sxx = sum(w * (u - mx) ** 2 for w, u in zip(weights, us))
    if n < 2 or sxx <= 1e-12:
        a, b = my, 0.0
    else:
        sxy = sum(w * (u - mx) * (y - my) for w, u, y in zip(weights, us, ys))
        b = sxy / sxx
        a = my - b * mx
    if b < 0.0:
        b = 0.0
        a = my
    rms = math.sqrt(sum(w * (y - (a + b * u)) ** 2
                        for w, u, y in zip(weights, us, ys)) / sw)
    n_eff = (sw * sw) / sum(w * w for w in weights)
    ub = mx
    sxx_lev = sxx
    sse = sum((y - (a + b * u)) ** 2 for u, y in zip(us, ys))
    return {"T": T, "a": a, "b": b, "rms": rms, "n": n,
            "n_eff": n_eff, "ubar": ub, "sxx": sxx_lev, "sse": sse}


def _model_weights(models, uniform_mix, n_eff):
    """AIC-style weights over candidate T values, with a uniform floor."""
    k = len(models)
    if k == 0:
        return []
    if n_eff >= 3:
        scores = []
        for m in models:
            sse = max(m["sse"], 1e-10)
            scores.append(n_eff * math.log(sse / n_eff) + 4.0)
        best = min(scores)
        ws = [math.exp(-0.5 * (s - best)) for s in scores]
        tot = sum(ws)
        ws = [w / tot for w in ws]
        ws = [(1.0 - uniform_mix) * w + uniform_mix / k for w in ws]
        tot = sum(ws)
        ws = [w / tot for w in ws]
    else:
        ws = [1.0 / k] * k
    return ws


def _sat(T, x):
    return 1.0 - math.exp(-x / T)


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------

def forecast_case(case, cfg=None):
    cfg = _cfg(cfg)
    trophies = case.get("trophies") or []
    n_tr = len(trophies)
    levels = case.get("levels") or LEVELS
    zvals = [norm_ppf(level) for level in levels]
    horizon_min = float(case.get("horizon_minutes") or 0.0)
    horizon_h = max(horizon_min, 0.0) / 60.0
    horizon_days = horizon_h / 24.0
    if cfg["short_floor"] >= 1.0 or horizon_h >= cfg["short_h0"]:
        between_damp = 1.0
    else:
        between_damp = max(cfg["short_floor"],
                           (max(horizon_h, 1e-6) / cfg["short_h0"]) ** cfg["short_pow"])

    game = case.get("game") or {}
    release = parse_time(game.get("release_date"))
    if release is None:
        release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case.get("target"))
    at_dt = parse_time(case.get("at"))

    states = distinct_states(case.get("history") or [])
    if not states:
        return {"trophies": [
            {"id": t.get("id"),
             "q": [round4(inv_logit_pct(z * 1.4)) for z in zvals]}
            for t in trophies]}

    def hours(dt):
        if dt is None:
            return 0.0
        return max((dt - release).total_seconds() / 3600.0, 0.0)

    ts = [hours(parse_time(st.get("at"))) for st in states]
    t_target = hours(target) if target is not None else (ts[-1] if ts else 0.0)
    t_last = ts[-1] if ts else 0.0
    span_h = max(0.0, ts[-1] - ts[0]) if ts else 0.0

    # ---- refresh cadence -------------------------------------------------
    state_times = [parse_time(st.get("at")) for st in states]
    gaps = []
    for i in range(1, len(state_times)):
        if state_times[i] is not None and state_times[i - 1] is not None:
            gaps.append((state_times[i] - state_times[i - 1]).total_seconds() / 60.0)
    gap_med = median_of(gaps) if gaps else 240.0
    gap_med = max(gap_med, 30.0)
    expected_refreshes = max(0.0, horizon_min / gap_med)
    t_cap_active = (cfg["t_span_mult"] > 0
                    and horizon_h >= cfg["t_cap_horizon_min"]
                    and (len(states) <= cfg["t_cap_states"]
                         or gap_med >= cfg["t_cap_gap_min"]))

    # ---- per-trophy last values -----------------------------------------
    last_rate = []
    last_E = []
    last_P_state = []
    for i in range(n_tr):
        rate = None
        for k in range(len(states) - 1, -1, -1):
            rr = states[k].get("rates") or []
            if i < len(rr) and rr[i] is not None:
                rate = float(rr[i])
                last_P_state.append(states[k].get("players_tracked"))
                break
        if rate is None:
            last_P_state.append(None)
            last_rate.append(1.0)
            last_E.append(0.5)
            continue
        P_here = last_P_state[-1]
        if P_here is None:
            P_here = None
            for k in range(len(states) - 1, -1, -1):
                if states[k].get("players_tracked"):
                    P_here = states[k].get("players_tracked")
                    break
        last_rate.append(rate)
        last_E.append(max(0.5, rate * float(P_here or 1.0) / 100.0))

    # ---- players model ---------------------------------------------------
    p_xs = []
    p_ys = []
    p_last = None
    for k, st in enumerate(states):
        P = st.get("players_tracked")
        if P is None:
            continue
        P = float(P)
        if P <= 0:
            continue
        p_xs.append(ts[k])
        p_ys.append(math.log(P))
        p_last = P
    if p_last is None:
        p_last = 1.0
    p_log_growth = (p_ys[-1] - p_ys[0]) if len(p_ys) >= 2 else 0.0
    if cfg["t_cap_log_growth"] > 0:
        t_cap_active = t_cap_active and p_log_growth >= cfg["t_cap_log_growth"]

    unique_p = len(set(round(math.exp(y), 4) for y in p_ys))
    p_flat = unique_p < 2 or (max(p_ys) - min(p_ys) < 0.02) if p_ys else True

    p_horizon = cfg["p_horizon_sigma"] * math.sqrt(max(horizon_days, 0.0))
    p_grid = list(cfg["p_T_grid"])
    if t_cap_active:
        t_cap = max(cfg["t_span_min"], cfg["t_span_mult"] * span_h)
        p_grid = [T for T in p_grid if T <= t_cap] or [min(p_grid)]
    if not p_flat and len(p_xs) >= 3:
        if cfg["p_half_life"] > 0:
            p_w = [0.5 ** ((p_xs[-1] - x) / cfg["p_half_life"]) for x in p_xs]
        else:
            p_w = None
        p_models = []
        for T in p_grid:
            m = _fit_one(p_xs, p_ys, T, p_w)
            if m is not None:
                p_models.append(m)
        ws = _model_weights(p_models, cfg["model_uniform"], len(p_xs))
        # each model has its own T
        yhat = 0.0
        within = 0.0
        preds = []
        for m, w in zip(p_models, ws):
            u = _sat(m["T"], t_target)
            p = m["a"] + m["b"] * u
            preds.append(p)
            yhat += w * p
            lev = 1.0 / m["n_eff"]
            if m["sxx"] > 1e-12:
                lev += (u - m["ubar"]) ** 2 / m["sxx"]
            within += w * (m["rms"] ** 2) * (1.0 + lev)
        between = sum(w * (p - yhat) ** 2 for p, w in zip(preds, ws))
        var_p = within + cfg["between_scale"] * between_damp * between
        yhat_p = max(yhat, math.log(p_last))
        sig_p = math.sqrt(max(var_p, 0.0) + p_horizon ** 2 + cfg["p_sigma_floor"] ** 2)
        sig_p = min(max(sig_p, cfg["p_sigma_floor"]), cfg["p_sigma_cap"])
    else:
        sd = math.sqrt(max(horizon_days, 0.0))
        yhat_p = math.log(p_last) + cfg["flat_p_drift"] * sd
        sig_p = cfg["flat_p_sigma0"] + cfg["flat_p_sigma"] * sd
        sig_p = math.sqrt(sig_p ** 2 + p_horizon ** 2)
        sig_p = min(max(sig_p, cfg["p_sigma_floor"]), cfg["flat_p_cap"])

    p_hat = math.exp(yhat_p)

    # ---- earners models --------------------------------------------------
    # Each read gives (t, log E, log P).  Earners are coupled to the player
    # base by an elasticity beta:  log E = y + beta * log P.
    #   beta = 1  -> new tracked players bring proportional earners (rate
    #                is driven only by the slow completion component y);
    #   beta = 0  -> earners are fixed while players grow (full dilution).
    # beta is estimated from consecutive refresh increments and shrunk to
    # the tier pool.  This is the structural part the direct-rate curve
    # misses: a 3x growth of the denominator need not move the rate if the
    # new cohort owns the trophy at the same rate.
    e_pts = []
    for i in range(n_tr):
        pts = []
        for k, st in enumerate(states):
            rr = st.get("rates") or []
            P = st.get("players_tracked")
            if P is None or P <= 0:
                continue
            if i < len(rr) and rr[i] is not None:
                rate = float(rr[i])
                E = rate * float(P) / 100.0
                if E >= cfg["e_min_count"]:
                    pts.append((ts[k], math.log(E), math.log(float(P))))
        e_pts.append(pts)

    # --- elasticity beta --------------------------------------------------
    beta_raw = []
    beta_pairs = []
    beta_se_raw = []
    beta_ok = []
    for pts in e_pts:
        num = den = 0.0
        diffs = []
        for k in range(1, len(pts)):
            dlp = pts[k][2] - pts[k - 1][2]
            if dlp <= 1e-4:
                continue
            dle = pts[k][1] - pts[k - 1][1]
            num += dle * dlp
            den += dlp * dlp
            diffs.append((dle, dlp))
        if den > 1e-8 and len(diffs) >= 2:
            b = num / den
            b = min(max(b, cfg["beta_min"]), cfg["beta_max"])
            resid = [dle - b * dlp for dle, dlp in diffs]
            rms = math.sqrt(sum(x * x for x in resid) / max(len(resid) - 1, 1))
            se = rms / math.sqrt(den) if den > 0 else 1.0
            beta_raw.append(b)
            beta_pairs.append(len(diffs))
            beta_se_raw.append(se)
            beta_ok.append(True)
        else:
            beta_raw.append(cfg["beta_prior"])
            beta_pairs.append(0)
            beta_se_raw.append(cfg["beta_sigma_floor"])
            beta_ok.append(False)

    tiers = [(t.get("tier") or "") for t in trophies]

    def _wmean(vals, wts):
        sw = sum(wts)
        if sw <= 0:
            return None
        return sum(v * w for v, w in zip(vals, wts)) / sw

    beta_pool = {}
    for tier in set(tiers) | {"__global__"}:
        vals = []
        wts = []
        ses = []
        for i in range(n_tr):
            if tier != "__global__" and tiers[i] != tier:
                continue
            if not beta_ok[i]:
                continue
            vals.append(beta_raw[i])
            wts.append(beta_pairs[i])
            ses.append(beta_se_raw[i])
        if vals:
            beta_pool[tier] = (_wmean(vals, wts),
                               median_of(ses) if ses else cfg["beta_sigma_floor"])
        else:
            beta_pool[tier] = None

    betas = []
    beta_ses = []
    for i in range(n_tr):
        pool = beta_pool.get(tiers[i]) or beta_pool.get("__global__")
        if not beta_ok[i]:
            if pool is not None:
                betas.append(pool[0])
                beta_ses.append(min(cfg["beta_no_data_sigma"],
                                    pool[1] + cfg["beta_sigma_floor"]))
            else:
                betas.append(cfg["beta_prior"])
                beta_ses.append(cfg["beta_no_data_sigma"])
            continue
        if pool is None:
            pool = (cfg["beta_prior"], cfg["beta_sigma_floor"])
        n = beta_pairs[i]
        w = n / (n + cfg["beta_shrink_k"])
        b = w * beta_raw[i] + (1.0 - w) * pool[0]
        b = (1.0 - cfg["beta_to_one"]) * b + cfg["beta_to_one"]
        se = math.sqrt((w * beta_se_raw[i]) ** 2
                       + ((1.0 - w) * pool[1]) ** 2
                       + cfg["beta_sigma_floor"] ** 2)
        betas.append(min(max(b, cfg["beta_min"]), cfg["beta_max"]))
        beta_ses.append(min(se, cfg["beta_sigma_cap"]))

    if cfg["beta_override"] is not None:
        betas = [float(cfg["beta_override"])] * n_tr
        beta_ses = [0.0] * n_tr

    # --- completion component y = log E - beta log P ----------------------
    y_pts = []
    for i in range(n_tr):
        y_pts.append([(t, le - betas[i] * lp) for t, le, lp in e_pts[i]])

    grid = list(cfg["e_T_grid"])
    if t_cap_active:
        t_cap = max(cfg["t_span_min"], cfg["t_span_mult"] * span_h)
        grid = [T for T in grid if T <= t_cap] or [min(grid)]

    # fits[T][i]
    fits = {T: [None] * n_tr for T in grid}
    for i in range(n_tr):
        pts = y_pts[i]
        if len(pts) < cfg["e_use_min"]:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if cfg["e_half_life"] > 0:
            t_end = xs[-1]
            ww = [0.5 ** ((t_end - x) / cfg["e_half_life"]) for x in xs]
        else:
            ww = None
        for T in grid:
            fits[T][i] = _fit_one(xs, ys, T, ww)

    # pooled slopes per tier and T
    pool = {}
    for T in grid:
        by_tier = {}
        allb = []
        for i in range(n_tr):
            m = fits[T][i]
            if m is None:
                continue
            if last_E[i] >= cfg["e_reliable"]:
                by_tier.setdefault(tiers[i], []).append(m["b"])
                allb.append(m["b"])
        g = sum(allb) / len(allb) if allb else 0.0
        pool[T] = {tier: (sum(v) / len(v)) for tier, v in by_tier.items()}
        pool[T]["__global__"] = g

    # tier-level model weights from pooled SSE of reliable trophies
    tier_weights = {}
    tier_keys = set(tiers) | {"__global__"}
    for tier in tier_keys:
        scores = []
        for T in grid:
            sse = 0.0
            cnt = 0
            for i in range(n_tr):
                if tier != "__global__" and tiers[i] != tier:
                    continue
                m = fits[T][i]
                if m is None or last_E[i] < cfg["e_reliable"]:
                    continue
                sse += m["sse"]
                cnt += m["n"]
            if cnt:
                scores.append((T, cnt * math.log(max(sse / cnt, 1e-10)) + 4.0))
            else:
                scores.append((T, None))
        valid = [(T, s) for T, s in scores if s is not None]
        if valid:
            best = min(s for _, s in valid)
            raw = {T: (math.exp(-0.5 * (s - best)) if s is not None else 0.0)
                   for T, s in scores}
            tot = sum(raw.values())
            k = len(scores)
            w = {T: (1.0 - cfg["model_uniform"]) * raw[T] / tot
                 + cfg["model_uniform"] / k for T in grid}
            tt = sum(w.values())
            tier_weights[tier] = {T: w[T] / tt for T in grid}
        else:
            tier_weights[tier] = {T: 1.0 / len(grid) for T in grid}

    med_logits = []
    sig_es = []
    r_hats = []
    e_parts = []
    for i in range(n_tr):
        tier = tiers[i]
        tw = tier_weights.get(tier) or tier_weights["__global__"]
        beta = betas[i]
        lp_last = math.log(max(last_P_state[i] or p_last, 1.0))
        yhats = []
        vars_ = []
        wts = []
        for T in grid:
            m = fits[T][i]
            if m is None:
                continue
            b = m["b"]
            n = m["n"]
            w_sh = n / (n + cfg["shrink_k"])
            pb = pool[T].get(tier)
            if pb is None:
                pb = pool[T]["__global__"]
            b_sh = w_sh * b + (1.0 - w_sh) * pb
            uL = _sat(T, ts[-1])
            uT = _sat(T, t_target)
            fitted_last = m["a"] + b * uL
            obs_last = math.log(max(last_E[i], 0.5)) - beta * lp_last
            anchor = cfg["e_anchor_obs"] * obs_last + \
                (1.0 - cfg["e_anchor_obs"]) * fitted_last
            yhat = anchor + b_sh * (uT - uL)
            lev = 1.0 / m["n_eff"]
            if m["sxx"] > 1e-12:
                lev += (uT - m["ubar"]) ** 2 / m["sxx"]
            var = (m["rms"] ** 2) * (1.0 + lev)
            yhats.append(yhat)
            vars_.append(var)
            wts.append(tw.get(T, 1.0 / len(grid)))
        if yhats:
            tot = sum(wts)
            wts = [w / tot for w in wts]
            yhat_y = sum(w * y for w, y in zip(wts, yhats))
            within = sum(w * v for w, v in zip(wts, vars_))
            between = sum(w * (y - yhat_y) ** 2 for w, y in zip(wts, yhats))
            var_y = within + cfg["between_scale"] * between_damp * between
        else:
            yhat_y = math.log(max(last_E[i], 0.5)) - beta * lp_last
            var_y = cfg["jump_prior"] ** 2
            within = var_y
            between = 0.0

        # quantisation noise of a small integer earner count
        e_now = max(last_E[i], 0.5)
        var_q = (cfg["e_quant_sigma"] ** 2) / max(e_now, 1.0)
        var_h = (cfg["e_horizon_sigma"] * math.sqrt(max(horizon_days, 0.0))) ** 2
        sig_y = math.sqrt(max(var_y, 0.0) + var_q + var_h)
        e_parts.append((within, cfg["between_scale"] * between_damp * between,
                        var_q + var_h))

        # implied rate and logit
        lp_T = yhat_p
        r = math.exp(min(yhat_y - (1.0 - beta) * lp_T, 50.0))
        r = min(max(r, 1e-9), 0.9995)
        mu = yhat_y - (1.0 - beta) * lp_T - math.log(1.0 - r)
        dlp = lp_T - lp_last
        beta_term = (abs(1.0 - beta) * sig_p + beta_ses[i] * abs(dlp))
        beta_term = min(beta_term, cfg["beta_term_cap"])
        inv = 1.0 / max(1.0 - r, 1e-4)
        sig = math.sqrt(sig_y ** 2 + (beta_term * inv) ** 2)
        med_logits.append(mu)
        sig_es.append(sig)
        r_hats.append(r)

    # ---- refresh jump scale ---------------------------------------------
    # Only the recent regime is informative about the noise at the target:
    # early-launch jumps are huge but have already died out.  Remove each
    # trophy's own recent mean jump (trend) before taking the RMS.
    recent = states[-min(len(states), cfg["jump_window"]):]
    per_tr_jumps = [[] for _ in range(n_tr)]
    for k in range(1, len(recent)):
        rp = recent[k - 1].get("rates") or []
        rc = recent[k].get("rates") or []
        for i in range(min(n_tr, len(rp), len(rc))):
            if rp[i] is None or rc[i] is None:
                continue
            per_tr_jumps[i].append(logit(float(rc[i])) - logit(float(rp[i])))
    resid = []
    for lst in per_tr_jumps:
        if len(lst) >= 2:
            m = sum(lst) / len(lst)
            resid.extend(j - m for j in lst)
        elif lst:
            resid.extend(lst)
    if resid:
        jump_pooled = math.sqrt(sum(j * j for j in resid) / len(resid))
    else:
        jump_pooled = cfg["jump_prior"]
    jump_pooled = min(jump_pooled, cfg["jump_prior"])

    # per-trophy noise scale, shrunk to the pooled value when the recent
    # sample is short (rare trophies have large integer-count jumps)
    jump_scale = []
    for lst in per_tr_jumps:
        if len(lst) >= 4:
            m = sum(lst) / len(lst)
            rms = math.sqrt(sum((j - m) ** 2 for j in lst) / len(lst))
            w = len(lst) / (len(lst) + cfg["jump_shrink_k"])
            jump_scale.append(min(w * rms + (1.0 - w) * jump_pooled,
                                  cfg["jump_prior"]))
        else:
            jump_scale.append(jump_pooled)

    jump_base = cfg["jump_mult"] * math.sqrt(
        min(expected_refreshes, cfg["jump_cap"]))
    s_jumps = [jump_base * j for j in jump_scale]
    s_rw = cfg["diffusion"] * math.sqrt(max(horizon_days, 0.0))

    rows = []
    diag_rows = []
    for i, trophy in enumerate(trophies):
        mu = med_logits[i]
        r = r_hats[i]
        # earner uncertainty + shared player-level (denominator) shock
        sig = math.sqrt(sig_es[i] ** 2
                        + s_jumps[i] ** 2 + s_rw ** 2)
        sig *= cfg["sigma_scale"]
        sig = min(max(sig, cfg["sigma_min"]), cfg["sigma_max"])

        # locked / near-100 % trophies
        if last_rate[i] >= cfg["locked_eps"] or r >= 0.999:
            mu = logit(99.99)
            sig = cfg["locked_sigma"]

        qs = []
        prev = -1.0
        asym_lo = cfg["asym_flat_lo"] if p_flat else (
            cfg["asym_cap_lo"] if t_cap_active else cfg["asym_lo"])
        asym_hi = cfg["asym_flat_hi"] if p_flat else (
            cfg["asym_cap_hi"] if t_cap_active else cfg["asym_hi"])
        for z in zvals:
            scale = sig * (asym_lo if z < 0 else asym_hi)
            q = round4(inv_logit_pct(mu + z * scale))
            q = min(100.0, max(0.0, q))
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy.get("id"), "q": qs})
        diag_rows.append({
            "id": trophy.get("id"), "mu": mu, "sig": sig,
            "sig_e": sig_es[i], "sig_p": sig_p, "r": r,
            "beta": betas[i],
            "s_jump": s_jumps[i], "s_rw": s_rw,
            "v_within": e_parts[i][0], "v_between": e_parts[i][1],
            "v_qh": e_parts[i][2],
        })
    if cfg.get("diag"):
        return {"trophies": rows, "_diag": diag_rows}
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
