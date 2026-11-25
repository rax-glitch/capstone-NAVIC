"""
NavIC/IRNSS L5 + S-band Ionospheric & Scintillation Workbench  (v2 - honest-data rewrite)

Design rules
  * Nothing is fabricated. If a quantity cannot be computed from the CSVs it shows "n/a".
  * Every anomaly model is a real, causal algorithm (no dead selectors).
  * sTEC is NOT clipped, abs()'d or forced to a fixed median.

Install:  pip install streamlit pandas numpy plotly numba
Run:      streamlit run app.py
"""
import io
import os
import re
import json
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*a, **k):
        return lambda f: f

st.set_page_config(page_title="NavIC Ionospheric Workbench", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
 .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
 div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; color: #38BDF8; }
 div[data-testid="stMetricLabel"] { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
F_L5, F_S, K_IONO, R_E = 1176.45e6, 2492.028e6, 40.3, 6371.0
# metres of (P_L5 - P_S) per TECU.  L5 is the lower frequency -> larger delay -> positive.
M_PER_TECU = K_IONO * (1.0 / F_L5**2 - 1.0 / F_S**2) * 1e16          # ~0.2263 m/TECU

M_Z = "Rolling Robust Z-Score (median/MAD baseline)"
M_KF = "Kalman Local-Linear-Trend (state-space innovation)"
M_DELTA = "Delta-Rule Online Predictor (DeltaNet-style, untrained)"
M_ROTI = "ROTI Irregularity Classifier (rolling baseline)"
MODELS = [M_Z, M_KF, M_DELTA, M_ROTI]

# ----------------------------------------------------------------------------
# Column detection (anchored regexes - no more "EL" matching "CHANNEL")
# ----------------------------------------------------------------------------
_SEP = r"[\s_\-]*"
PER_CH = {
    "pr":   r"^PR" + _SEP + "{ch}(?!\\d)",
    "lock": r"^Lock" + _SEP + r"(Time)?" + _SEP + "{ch}(?!\\d)",
    "cn0":  r"^(CN0|C/N0|CNO|SNR)" + _SEP + "{ch}(?!\\d)",
    "el":   r"^(El|Elev|Elevation)" + _SEP + "{ch}(?!\\d)",
    "az":   r"^(Az|Azim|Azimuth)" + _SEP + "{ch}(?!\\d)",
}
GENERIC = {
    "cn0": r"^(CN0|C/N0|CNO|SNR)(\s*\(.*\))?$",
    "el":  r"^(El|Elev|Elevation)(\s*\(.*\))?$",
    "az":  r"^(Az|Azim|Azimuth)(\s*\(.*\))?$",
    "lat": r"^(Lat|Latitude)(\s*\(.*\))?$",
    "lon": r"^(Lon|Long|Longitude)(\s*\(.*\))?$",
    "alt": r"^(Alt|Altitude|Height)(\s*\(.*\))?$",
    "t":   r"^(TOWC|TOW|TOWS)(\s*\(.*\))?$",
}


def find_col(cols, key, ch=None, generic=True):
    if ch is not None and key in PER_CH:
        rx = re.compile(PER_CH[key].replace("{ch}", str(ch)), re.I)
        for c in cols:
            if rx.search(c):
                return c
    if generic and key in GENERIC:
        rx = re.compile(GENERIC[key], re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None


def detect_channels(df):
    out = set()
    for c in df.columns:
        m = re.match(r"^PR[\s_\-]*(\d+)", c, re.I)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def clean_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ----------------------------------------------------------------------------
# Cached loaders (58 MB files are read ONCE, not on every widget click)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading CSV...")
def load_path(path, mtime, size):
    return clean_cols(pd.read_csv(path, low_memory=False))


@st.cache_data(show_spinner="Reading CSV...")
def load_upload(_raw, key):
    return clean_cols(pd.read_csv(io.BytesIO(_raw), low_memory=False))


def stretch(fn, obj, **kw):
    """Works on old and new Streamlit (use_container_width -> width='stretch')."""
    if "width" in inspect.signature(fn).parameters:
        return fn(obj, width="stretch", **kw)
    return fn(obj, use_container_width=True, **kw)


# ----------------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------------
def make_time(s, scale):
    """Seconds-of-week -> monotonic seconds (handles GPS-week rollover at 604800 s)."""
    t = s.astype(float) * scale
    return t + (t.diff() < -302400).cumsum() * 604800.0


def segments(arc):
    """(start, end) slices of contiguous valid arcs (arc id >= 0)."""
    if len(arc) == 0:
        return []
    ch = np.flatnonzero(np.diff(arc) != 0) + 1
    st_, en_ = np.r_[0, ch], np.r_[ch, len(arc)]
    return [(a, b) for a, b in zip(st_, en_) if arc[a] >= 0]


def rot_roti(t, x, dt_med):
    """ROT (TECU/min, 1-min differencing) and ROTI (std of ROT over 5 min), inside one arc."""
    rot = np.full(len(x), np.nan)
    roti = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return rot, roti
    tt, xx = t[ok], x[ok]
    prev = np.interp(tt - 60.0, tt, xx, left=np.nan, right=np.nan)
    rot[ok] = xx - prev                              # TECU per 60 s = TECU/min
    s = pd.Series(rot, index=pd.to_datetime(t, unit="s"))
    nmin = int(np.clip(0.5 * 300.0 / dt_med, 3, 150))
    roti = s.rolling("300s", min_periods=nmin).std().values
    return rot, roti


@njit(cache=True)
def kalman_pred(t, x, reset, r, qs):
    """One-step-ahead prediction of a local-linear-trend Kalman filter (level + slope)."""
    n = len(x)
    pred = np.full(n, np.nan)
    init = False
    L = 0.0; S = 0.0; p00 = 0.0; p01 = 0.0; p11 = 0.0
    for i in range(n):
        xi = x[i]
        if np.isnan(xi):
            init = False
            continue
        if reset[i] or (not init):
            L = xi; S = 0.0; p00 = r; p01 = 0.0; p11 = r * 1e-2
            init = True
            continue
        dt = t[i] - t[i - 1]
        Lp = L + S * dt
        a00 = p00 + 2.0 * dt * p01 + dt * dt * p11
        a01 = p01 + dt * p11
        a11 = p11 + qs * r * dt
        pred[i] = Lp
        v = xi - Lp
        sv = a00 + r
        k0 = a00 / sv
        k1 = a01 / sv
        L = Lp + k0 * v
        S = S + k1 * v
        p00 = (1.0 - k0) * a00
        p01 = (1.0 - k0) * a01
        p11 = a11 - k1 * a01
    return pred


@njit(cache=True)
def delta_pred(x, reset, beta, eta, d):
    """
    Online delta-rule (normalised-LMS with forget gate beta) predictor of the next
    first-difference from the last d differences - a causal, DeltaNet-style linear
    state update.  Weights learn online; there is no offline training.
    """
    n = len(x)
    pred = np.full(n, np.nan)
    W = np.zeros(d)
    hist = np.zeros(d)
    init = False
    prev = 0.0
    for i in range(n):
        xi = x[i]
        if np.isnan(xi):
            init = False
            continue
        if reset[i] or (not init):
            for j in range(d):
                hist[j] = 0.0
            prev = xi
            init = True
            continue
        dx = xi - prev
        p = 0.0
        nrm = 1e-3
        for j in range(d):
            p += W[j] * hist[j]
            nrm += hist[j] * hist[j]
        pred[i] = prev + p
        e = dx - p
        for j in range(d):
            W[j] = beta * W[j] + eta * e * hist[j] / nrm
        for j in range(d - 1, 0, -1):
            hist[j] = hist[j - 1]
        hist[0] = dx
        prev = xi
    return pred


def geometry_table(l5, tcol, tscale, channels, el_cut, min_lock):
    """Active-SV count and TRUE DOP from per-channel El/Az (needs >= 4 channels with El+Az)."""
    t = make_time(pd.to_numeric(l5[tcol], errors="coerce"), tscale).values
    n = len(t)
    cnt = np.zeros(n)
    any_chan = False
    EL, AZ, OK = [], [], []
    for c in channels:
        pr = find_col(l5.columns, "pr", c, generic=False)
        lk = find_col(l5.columns, "lock", c, generic=False)
        el = find_col(l5.columns, "el", c, generic=False)
        az = find_col(l5.columns, "az", c, generic=False)
        if pr is None:
            continue
        ok = np.isfinite(pd.to_numeric(l5[pr], errors="coerce").values)
        if lk is not None:
            ok &= pd.to_numeric(l5[lk], errors="coerce").values >= min_lock
        elv = pd.to_numeric(l5[el], errors="coerce").values if el else None
        if elv is not None:
            ok &= elv >= el_cut
        cnt += ok
        any_chan = True
        if el and az:
            EL.append(elv)
            AZ.append(pd.to_numeric(l5[az], errors="coerce").values)
            OK.append(ok)
    out = pd.DataFrame({"t": t, "Num_SV": cnt if any_chan else np.nan})
    for k in ("GDOP", "PDOP", "HDOP", "VDOP"):
        out[k] = np.nan
    if len(EL) >= 4:
        EL, AZ, OK = np.column_stack(EL), np.column_stack(AZ), np.column_stack(OK)
        OK = OK & np.isfinite(EL) & np.isfinite(AZ)
        res = np.full((n, 4), np.nan)
        for s0 in range(0, n, 100000):
            sl = slice(s0, min(s0 + 100000, n))
            el = np.deg2rad(np.nan_to_num(EL[sl])); az = np.deg2rad(np.nan_to_num(AZ[sl]))
            w = OK[sl].astype(float); ce = np.cos(el)
            H = np.stack([-ce * np.sin(az), -ce * np.cos(az), -np.sin(el), np.ones_like(el)], axis=-1)
            N = np.einsum("mci,mcj,mc->mij", H, H, w)
            good = (w.sum(1) >= 4) & (np.abs(np.linalg.det(N)) > 1e-9)
            if good.any():
                Q = np.linalg.inv(N[good])
                idx = np.flatnonzero(good) + s0
                res[idx, 0] = np.sqrt(np.trace(Q, axis1=1, axis2=2))
                res[idx, 1] = np.sqrt(Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2])
                res[idx, 2] = np.sqrt(Q[:, 0, 0] + Q[:, 1, 1])
                res[idx, 3] = np.sqrt(Q[:, 2, 2])
        out[["GDOP", "PDOP", "HDOP", "VDOP"]] = res
    out = out.dropna(subset=["t"]).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    return out, len(EL)


# ----------------------------------------------------------------------------
# Physics pipeline (cached; independent of model / threshold so sliders stay instant)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing sTEC, ROT/ROTI, S4 proxy, DOP ...")
def compute_physics(_l5, _s, data_key, cfg):
    m, notes = cfg["maps"], []

    def col(df, name):
        if name and name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    if not all(m[k] for k in ("t_l5", "t_s", "pr_l5", "pr_s")):
        return None, {"error": "Time and pseudorange columns must be mapped for both files."}

    a = pd.DataFrame({"t": make_time(col(_l5, m["t_l5"]), cfg["tscale"]),
                      "PR_L5": col(_l5, m["pr_l5"]), "Lock_L5": col(_l5, m["lock_l5"]),
                      "CN0_L5": col(_l5, m["cn0_l5"]), "El": col(_l5, m["el"]), "Az": col(_l5, m["az"]),
                      "LAT": col(_l5, m["lat"]), "LON": col(_l5, m["lon"]), "ALT": col(_l5, m["alt"])})
    b = pd.DataFrame({"t": make_time(col(_s, m["t_s"]), cfg["tscale"]),
                      "PR_S": col(_s, m["pr_s"]), "Lock_S": col(_s, m["lock_s"]),
                      "CN0_S": col(_s, m["cn0_s"])})
    a = a.dropna(subset=["t"]).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    b = b.dropna(subset=["t"]).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    df = pd.merge_asof(a, b, on="t", direction="nearest", tolerance=0.5)
    if df["PR_S"].notna().sum() < 10:
        return None, {"error": "L5 and S-band logs share almost no common epochs (check time columns/units)."}

    geom, n_geo = geometry_table(_l5, m["t_l5"], cfg["tscale"], cfg["channels"], cfg["el_cut"], cfg["min_lock"])
    df = pd.merge_asof(df, geom, on="t", direction="nearest", tolerance=0.5)
    if n_geo < 4:
        notes.append(f"DOP not computed: only {n_geo} channel(s) have both El and Az columns (need >= 4).")
    if df["Num_SV"].isna().all():
        notes.append("Active-SV count unavailable (no PR columns found per channel).")

    # ---- validity: finite PR, lock, elevation mask -------------------------------------
    valid = df["PR_L5"].notna().values & df["PR_S"].notna().values
    lock_cols = [c for c in ("Lock_L5", "Lock_S") if df[c].notna().any()]
    if lock_cols:
        for c in lock_cols:
            valid &= (df[c].values >= cfg["min_lock"])
    else:
        notes.append("No lock-time columns mapped: lock filtering NOT applied.")
    if df["El"].notna().any():
        valid &= (df["El"].values >= cfg["el_cut"])
    else:
        notes.append("No elevation column for this channel: elevation mask NOT applied, VTEC and sky plot unavailable.")

    t = df["t"].values
    # -- plausibility gates: sentinel/zero pseudoranges and physically impossible TEC --
    pr5, prs = df["PR_L5"].values, df["PR_S"].values
    n_before = int(valid.sum())
    valid &= (pr5 > 0) & (prs > 0)
    raw0 = (pr5 - prs) / M_PER_TECU
    if valid.any():
        med0 = float(np.nanmedian(raw0[valid]))
        valid &= np.abs(raw0 - med0) <= cfg["max_dev"]

    def build_arcs(valid):
        dt = np.diff(t, prepend=np.nan)
        prev_valid = np.r_[False, valid[:-1]]
        start = valid & ((~prev_valid) | (dt > cfg["gap_s"]) | np.isnan(dt))
        return np.where(valid, np.cumsum(start), -1).astype(np.int64)

    arc = build_arcs(valid)
    for a_, b_ in segments(arc):                       # drop arcs that are too short to trust
        if t[b_ - 1] - t[a_] < cfg["min_arc_s"]:
            valid[a_:b_] = False
    arc = build_arcs(valid)
    n_rejected = n_before - int(valid.sum())
    df["arc"], df["valid"] = arc, valid
    segs = segments(arc)
    if not segs:
        return None, {"error": "No valid epochs after lock/elevation filtering. Lower the mask or lock threshold."}
    same = (arc[1:] == arc[:-1]) & (arc[1:] >= 0)
    dts = np.diff(t)[same]
    dt_med = float(np.median(dts)) if len(dts) else float(np.nanmedian(np.diff(t)))
    dt_med = max(dt_med, 1e-3)

    # ---- sTEC from code difference (no abs, no clip, no forced median) -----------------
    raw = (df["PR_L5"].values - df["PR_S"].values) / M_PER_TECU
    raw = np.where(valid, raw, np.nan)
    med_raw = float(np.nanmedian(raw))
    if cfg["cal_mode"] == "Manual DCB (TECU)":
        tec = raw - cfg["dcb"]
    elif cfg["cal_mode"] == "Align arc median to reference TECU":
        tec = raw - (med_raw - cfg["ref_tec"])
        notes.append(f"Median forced to {cfg['ref_tec']} TECU by user choice - absolute level is an assumption, not a measurement.")
    else:
        tec = raw.copy()
        notes.append("sTEC is UNCALIBRATED (includes satellite+receiver DCB). Relative variations are valid; absolute level is not.")
    df["sTEC_raw"] = raw

    n_hamp = 0
    n_smooth = int(round(cfg["smooth_s"] / dt_med))
    for a_, b_ in segs:
        seg = pd.Series(tec[a_:b_])
        if cfg["hampel_k"] > 0 and len(seg) >= 8:
            med = seg.rolling(31, center=True, min_periods=5).median()
            dev = (seg - med).abs()
            mad = 1.4826 * dev.rolling(31, center=True, min_periods=5).median()
            bad = (dev > cfg["hampel_k"] * np.maximum(mad, 0.05)).values
            n_hamp += int(bad.sum())
            seg = seg.mask(bad)
        if n_smooth > 1:
            seg = seg.rolling(n_smooth, center=True, min_periods=max(1, n_smooth // 2)).median()
        tec[a_:b_] = seg.values
    df["sTEC"] = tec

    # ---- VTEC (thin shell) -----------------------------------------------------------
    if df["El"].notna().any():
        el = np.deg2rad(df["El"].values)
        mf = np.sqrt(np.clip(1.0 - (R_E * np.cos(el) / (R_E + cfg["h_ion"])) ** 2, 0.0, 1.0))
        df["VTEC"] = df["sTEC"] * mf
    else:
        df["VTEC"] = np.nan

    # ---- ROT / ROTI per arc (no clipping) --------------------------------------------
    rot = np.full(len(df), np.nan); roti = np.full(len(df), np.nan)
    for a_, b_ in segs:
        rot[a_:b_], roti[a_:b_] = rot_roti(t[a_:b_], tec[a_:b_], dt_med)
    df["ROT"], df["ROTI"] = rot, roti

    # ---- S4 PROXY from receiver-reported C/N0 ------------------------------------------
    n60 = max(5, int(round(60.0 / dt_med)))
    for band in ("L5", "S"):
        cn = df[f"CN0_{band}"].values
        s4 = np.full(len(df), np.nan)
        if np.isfinite(cn).any():
            for a_, b_ in segs:
                I = 10.0 ** (pd.Series(cn[a_:b_]) / 10.0)
                m1 = I.rolling(n60, min_periods=n60).mean()
                m2 = (I ** 2).rolling(n60, min_periods=n60).mean()
                s4[a_:b_] = (np.sqrt(np.maximum(m2 - m1 ** 2, 0.0)) / m1).values
        df[f"S4_{band}"] = s4
    if df["CN0_L5"].isna().all():
        notes.append("No L5 C/N0 column for this channel: S4 proxy unavailable.")
    else:
        notes.append("S4 is a PROXY (60 s, uncorrected) from 1-Hz-class C/N0; true S4 needs 50-100 Hz intensity.")

    # ---- Position scatter / error ------------------------------------------------------
    df["Pos_Err_H"] = np.nan; df["Pos_Err_3D"] = np.nan
    if df["LAT"].notna().any() and df["LON"].notna().any():
        ref = cfg["ref_pos"]
        lat0, lon0, alt0 = ref if ref else (float(np.nanmedian(df["LAT"])), float(np.nanmedian(df["LON"])),
                                            float(np.nanmedian(df["ALT"])) if df["ALT"].notna().any() else 0.0)
        dN = np.deg2rad(df["LAT"] - lat0) * 6371000.0
        dE = np.deg2rad(df["LON"] - lon0) * 6371000.0 * np.cos(np.deg2rad(lat0))
        dU = (df["ALT"] - alt0) if df["ALT"].notna().any() else 0.0
        df["Pos_Err_H"] = np.hypot(dN, dE)
        df["Pos_Err_3D"] = np.sqrt(df["Pos_Err_H"] ** 2 + dU ** 2)
        notes.append("Position error = " + ("distance from your reference position." if ref else
                     "scatter about the MEDIAN position (not true error unless a surveyed reference is given)."))
    else:
        notes.append("No LAT/LON columns: position error unavailable.")

    df["th"] = (df["t"] - df["t"].iloc[0]) / 3600.0
    info = {
        "notes": notes, "n_total": len(df), "n_valid": int(valid.sum()), "n_arcs": len(segs),
        "dt_med": dt_med, "n_hampel": n_hamp, "t0": float(df["t"].iloc[0]),
        "stec_median": float(np.nanmedian(tec)), "frac_neg": float(np.nanmean(tec < 0)),
        "dop_channels": n_geo, "n_rejected": n_rejected,
    }
    return df, info


# ----------------------------------------------------------------------------
# Anomaly models (all causal; score = |residual| / trailing robust sigma)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running anomaly model ...")
def compute_scores(_df, phys_key, model, win, beta, kf_logq):
    t, arc = _df["t"].values, _df["arc"].values
    x = (_df["ROTI"] if model == M_ROTI else _df["sTEC"]).values.astype(float)
    n, segs, mp = len(x), segments(arc), max(10, win // 5)
    reset = np.zeros(n, dtype=np.bool_)
    reset[0] = True
    reset[1:] = arc[1:] != arc[:-1]
    pred = np.full(n, np.nan)

    if model in (M_Z, M_ROTI):
        for a, b in segs:
            pred[a:b] = pd.Series(x[a:b]).rolling(win, min_periods=mp).median().shift(1).values
    elif model == M_KF:
        ok = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (arc[1:] == arc[:-1]) & (arc[1:] >= 0)
        d = (x[1:] - x[:-1])[ok]
        sig = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0) if len(d) > 10 else 1.0
        pred = kalman_pred(t, x, reset, max(sig ** 2, 1e-4), 10.0 ** kf_logq)
    elif model == M_DELTA:
        pred = delta_pred(np.ascontiguousarray(x), reset, float(beta), 0.3, 8)

    resid = x - pred
    sig = np.full(n, np.nan)
    for a, b in segs:
        sig[a:b] = 1.4826 * pd.Series(np.abs(resid[a:b])).rolling(win, min_periods=mp).median().shift(1).values
    fin = sig[np.isfinite(sig)]
    floor = max(0.1 * float(np.median(fin)), 1e-3) if len(fin) else 1e-3
    r_use = np.maximum(resid, 0.0) if model == M_ROTI else np.abs(resid)   # ROTI: only increases matter
    score = r_use / np.maximum(sig, floor)
    return pd.DataFrame({"Model_Pred": pred, "Model_Resid": resid, "Model_Score": score}, index=_df.index)


def sustained_mask(score, thr, n_req):
    """Flag epochs that belong to >= n_req consecutive epochs above thr. Returns (mask, peak sustained level)."""
    sc = pd.Series(score)
    level = sc.rolling(n_req, min_periods=n_req).min()
    run_end = ((sc > thr).astype(float).rolling(n_req, min_periods=n_req).min() == 1).astype(float)
    mask = run_end.iloc[::-1].rolling(n_req, min_periods=1).max().iloc[::-1] > 0
    return mask.values, (float(level.max()) if level.notna().any() else np.nan)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def make_plot_df(d, dt_med, gap_s, max_pts=60000):
    """Decimate but keep per-bucket extremes (spikes survive) and keep gaps as gaps."""
    n = len(d)
    k = 1
    if n > max_pts:
        k = int(np.ceil(n / max_pts))
        d = d.reset_index(drop=True)
        pos = np.arange(n); bucket = pos // k
        keep = set(pos[::k].tolist())
        for c, fn in (("Model_Score", "idxmax"), ("ROTI", "idxmax"), ("sTEC", "idxmax"), ("sTEC", "idxmin")):
            v = d[c].values.astype(float)
            v = np.where(np.isnan(v), -np.inf if fn == "idxmax" else np.inf, v)   # all-NaN bucket safe
            g = getattr(pd.Series(v).groupby(bucket), fn)()
            keep.update(g.astype(int).tolist())
        d = d.iloc[sorted(keep)]
    gap = max(gap_s, 3.0 * k * dt_med)
    gi = np.flatnonzero(np.diff(d["t"].values) > gap)
    if len(gi):
        filler = pd.DataFrame(np.nan, index=range(len(gi)), columns=d.columns)
        filler["th"] = d["th"].values[gi] + 1e-6
        d = pd.concat([d, filler]).sort_values("th", kind="mergesort")
    return d


def add_line(fig, d, col, name, color, row, width=1.2, dash=None, **kw):
    if col in d and d[col].notna().any():
        fig.add_trace(go.Scattergl(x=d["th"], y=d[col], name=name,
                                   line=dict(color=color, width=width, dash=dash), **kw), row=row, col=1)
        return True
    return False


def fmt(v, f):
    return "n/a" if v is None or not np.isfinite(v) else format(v, f)


def nmax(s):
    s = s.dropna()
    return float(s.max()) if len(s) else np.nan


def parse_ref(s):
    try:
        p = [float(x) for x in s.split(",")]
        return tuple(p) if len(p) == 3 else None
    except Exception:
        return None


def dark(fig, h, n_rows):
    fig.update_layout(template="plotly_dark", height=h, margin=dict(l=40, r=40, t=60, b=30),
                      legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="Elapsed time since first epoch (h)", row=n_rows, col=1)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🛰️ NavIC/IRNSS Dual-Frequency Ionospheric & Scintillation Workbench")
st.caption("L5 (1176.45 MHz) + S-band (2492.028 MHz) code-TEC, ROT/ROTI, S4 proxy, real DOP • causal anomaly models • no synthetic data")

df_l5 = df_s = None
data_key = None
with st.sidebar:
    st.header("🔬 1. Telemetry ingestion")
    mode = st.radio("Data source", ["Local disk / repo path (recommended)", "Browser upload (small files only)"])
    if mode.startswith("Local"):
        def first_existing(name):
            for p in (f"/{name}", f"/content/{name}", name):
                if os.path.exists(p):
                    return p
            return name
        l5_path = st.text_input("L5 log path", first_existing("SATB_L5 (1).csv"))
        s_path = st.text_input("S-band log path", first_existing("SATB_S (1).csv"))
        if os.path.exists(l5_path) and os.path.exists(s_path):
            df_l5 = load_path(l5_path, os.path.getmtime(l5_path), os.path.getsize(l5_path))
            df_s = load_path(s_path, os.path.getmtime(s_path), os.path.getsize(s_path))
            data_key = f"{l5_path}|{os.path.getmtime(l5_path)}|{s_path}|{os.path.getmtime(s_path)}"
            st.success(f"✅ Loaded {len(df_l5):,} L5 rows / {len(df_s):,} S rows")
        else:
            st.error("⚠️ File path not found.")
    else:
        up_l5 = st.file_uploader("L5 CSV", type=["csv"])
        up_s = st.file_uploader("S-band CSV", type=["csv"])
        if up_l5 and up_s:
            df_l5 = load_upload(up_l5.getvalue(), f"{up_l5.name}|{up_l5.size}")
            df_s = load_upload(up_s.getvalue(), f"{up_s.name}|{up_s.size}")
            data_key = f"{up_l5.name}|{up_l5.size}|{up_s.name}|{up_s.size}"

if df_l5 is None or df_s is None:
    st.info("👈 Point the sidebar at your L5 and S-band CSV files to start.")
    st.stop()

channels = detect_channels(df_l5)
if not channels:
    st.error("No pseudorange columns like `PR-1 (m)` found in the L5 file.")
    st.write("Columns found:", list(df_l5.columns))
    st.stop()

with st.sidebar:
    tscale = {"seconds": 1.0, "milliseconds": 1e-3}[st.selectbox("TOWC units", ["seconds", "milliseconds"])]
    st.divider()
    st.header("🛰️ 2. Channel & masks")
    ch = st.selectbox("Channel / PRN", channels)
    el_cut = st.slider("Elevation mask (°)", 0, 40, 15, help="Removes low-elevation multipath / long slant paths.")
    min_lock = st.number_input("Min lock time (s)", 0, 600, 30, 5)
    h_ion = st.number_input("Iono shell height (km)", 250, 450, 350, 10)

    with st.expander("🧩 Column mapping (auto-detected - override if wrong)"):
        def pick(label, df, default, key):
            opts = ["(none)"] + list(df.columns)
            v = st.selectbox(label, opts, index=opts.index(default) if default in opts else 0, key=f"{key}_{ch}")
            return None if v == "(none)" else v
        c5, cs = df_l5.columns, df_s.columns
        maps = {
            "t_l5": pick("L5 time", df_l5, find_col(c5, "t"), "t5"),
            "t_s": pick("S time", df_s, find_col(cs, "t"), "ts"),
            "pr_l5": pick("L5 pseudorange", df_l5, find_col(c5, "pr", ch, False), "pr5"),
            "pr_s": pick("S pseudorange", df_s, find_col(cs, "pr", ch, False), "prs"),
            "lock_l5": pick("L5 lock time", df_l5, find_col(c5, "lock", ch, False), "lk5"),
            "lock_s": pick("S lock time", df_s, find_col(cs, "lock", ch, False), "lks"),
            "cn0_l5": pick("L5 C/N0", df_l5, find_col(c5, "cn0", ch), "cn5"),
            "cn0_s": pick("S C/N0", df_s, find_col(cs, "cn0", ch), "cns"),
            "el": pick("Elevation", df_l5, find_col(c5, "el", ch), "el"),
            "az": pick("Azimuth", df_l5, find_col(c5, "az", ch), "az"),
            "lat": pick("Latitude", df_l5, find_col(c5, "lat"), "lat"),
            "lon": pick("Longitude", df_l5, find_col(c5, "lon"), "lon"),
            "alt": pick("Altitude", df_l5, find_col(c5, "alt"), "alt"),
        }
    ref_txt = st.text_input("Reference position lat,lon,alt (optional)", "", help="Blank = scatter about median position.")

    st.divider()
    st.header("🧪 3. TEC extraction")
    cal_mode = st.selectbox("DCB handling", ["None (raw, relative TEC)", "Manual DCB (TECU)", "Align arc median to reference TECU"])
    dcb = st.number_input("DCB to subtract (TECU)", value=0.0, step=1.0) if cal_mode.startswith("Manual") else 0.0
    ref_tec = st.number_input("Reference median TECU", value=20.0, step=1.0) if cal_mode.startswith("Align") else 20.0
    gap_s = st.number_input("Max gap inside an arc (s)", 1.0, 600.0, 10.0, 1.0)
    smooth_s = st.number_input("Code-noise smoothing (s, 0 = off)", 0.0, 120.0, 15.0, 5.0,
                               help="Rolling median. Code TEC is noisy (~0.2 m per TECU); smoothing trades speed for noise.")
    max_dev = st.number_input("Reject |sTEC − median| larger than (TECU)", 20.0, 2000.0, 200.0, 10.0,
                              help="Removes sentinel values / clock-offset garbage. Real TEC rarely moves >200 TECU from its median.")
    min_arc_s = st.number_input("Min arc length (s)", 0.0, 3600.0, 120.0, 30.0,
                                help="Arcs shorter than this are dropped (too short for despiking/ROTI).")
    hampel_k = st.number_input("Outlier/cycle-slip despike k·MAD (0 = off)", 0.0, 10.0, 5.0, 0.5)

    st.divider()
    st.header("🧠 4. Anomaly detection")
    model = st.selectbox("Model", MODELS)
    win = st.number_input("Baseline window (epochs)", 30, 3600, 300, 30)
    thr = st.slider("Trigger threshold (σ)", 1.0, 8.0, 3.0, 0.1)
    min_dur = st.number_input("Must persist for (s)", 1.0, 600.0, 10.0, 1.0)
    beta = st.slider("Delta-rule forget gate β", 0.80, 0.999, 0.99, 0.001) if model == M_DELTA else 0.99
    kf_logq = st.slider("Kalman process-noise ratio log10(q/r)", -7.0, -2.0, -4.0, 0.5) if model == M_KF else -4.0

cfg = {"ch": int(ch), "maps": maps, "h_ion": float(h_ion), "cal_mode": cal_mode, "dcb": float(dcb),
       "ref_tec": float(ref_tec), "el_cut": float(el_cut), "gap_s": float(gap_s), "smooth_s": float(smooth_s),
       "hampel_k": float(hampel_k), "max_dev": float(max_dev), "min_arc_s": float(min_arc_s), "tscale": tscale, "min_lock": float(min_lock),
       "ref_pos": parse_ref(ref_txt), "channels": tuple(channels)}

df, info = compute_physics(df_l5, df_s, data_key, cfg)
if df is None:
    st.error(info["error"])
    st.stop()
if not HAS_NUMBA and len(df) > 200_000:
    st.warning("numba is not installed - Kalman/Delta models will be slow on large logs. `pip install numba`.")

phys_key = (data_key, json.dumps(cfg, sort_keys=True, default=str))
df = pd.concat([df, compute_scores(df, phys_key, model, int(win), float(beta), float(kf_logq))], axis=1)
n_req = max(1, int(round(min_dur / info["dt_med"])))
mask, peak_sustained_all = sustained_mask(df["Model_Score"].values, thr, n_req)
df["Trigger"] = mask

with st.expander("📋 Data-quality report & assumptions", expanded=False):
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Epochs (merged)", f"{info['n_total']:,}")
    q2.metric("Valid epochs", f"{info['n_valid']:,}")
    q3.metric("Continuous arcs", info["n_arcs"])
    q4.metric("Sample interval", f"{info['dt_med']:.2f} s")
    st.write(f"Rejected by plausibility/short-arc gates: **{info['n_rejected']:,}** epochs • Despiked outliers: **{info['n_hampel']:,}** • median sTEC: **{info['stec_median']:.1f} TECU** • "
             f"negative sTEC: **{100 * info['frac_neg']:.1f}%** (large share ⇒ DCB offset not removed)")
    for n_ in info["notes"]:
        st.warning(n_)

span = float(df["th"].max())
if span < 0.02:
    st.error("Selected data spans < ~1 minute; nothing to plot.")
    st.stop()
rng = st.slider("⏱️ Time range (hours since first epoch)", 0.0, span, (0.0, span), step=max(span / 2000, 0.005))
view = df[(df["th"] >= rng[0]) & (df["th"] <= rng[1])]
if view.empty:
    st.warning("Empty selection.")
    st.stop()

v_mask = view["Trigger"].values
peak_raw = nmax(view["Model_Score"])
peak_sus = nmax(pd.Series(view["Model_Score"].values).rolling(n_req, min_periods=n_req).min())
is_anom = bool(v_mask.any())

m = st.columns(6)
m[0].metric("Peak sustained score", f"{fmt(peak_sus, '.2f')} σ", help=f"Highest level held for ≥ {min_dur:.0f}s. Raw single-epoch peak: {fmt(peak_raw, '.2f')} σ")
m[1].metric("Peak sTEC / VTEC", f"{fmt(nmax(view['sTEC']), '.1f')} / {fmt(nmax(view['VTEC']), '.1f')} TECU")
m[2].metric("Max ROTI (5 min)", f"{fmt(nmax(view['ROTI']), '.3f')} TECU/min")
m[3].metric("Peak L5 S4 (proxy)", fmt(nmax(view["S4_L5"]), ".3f"))
m[4].metric("Mean PDOP", fmt(float(view["PDOP"].mean()) if view["PDOP"].notna().any() else np.nan, ".2f"))
m[5].metric("Peak 3D pos scatter", f"{fmt(nmax(view['Pos_Err_3D']), '.2f')} m")

if is_anom:
    st.error(f"🚨 **ANOMALY:** score > {thr}σ sustained ≥ {min_dur:.0f}s in {int(v_mask.sum()):,} epochs "
             f"({100 * v_mask.mean():.1f}% of view).")
else:
    st.success(f"✅ **Nominal:** no sustained excursion above {thr}σ in this window.")

dp = make_plot_df(view, info["dt_med"], gap_s)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.6, 0.4],
                    subplot_titles=(f"sTEC vs model one-step prediction (PRN {ch}, {cal_mode.split(' (')[0]})",
                                    f"Anomaly score (threshold {thr}σ, shaded markers = sustained trigger)"))
add_line(fig, dp, "sTEC", "sTEC (TECU)", "#38BDF8", 1, 1.2)
add_line(fig, dp, "Model_Pred", "Model prediction", "#F59E0B", 1, 1.3, "dot")
add_line(fig, dp, "Model_Score", "Score", "#EC4899", 2, 1.1)
fig.add_hline(y=thr, line_dash="dash", line_color="#EF4444", annotation_text=f"{thr}σ", row=2, col=1)
trig = view[view["Trigger"]]
if len(trig):
    trig = trig.iloc[:: max(1, len(trig) // 4000)]
    fig.add_trace(go.Scattergl(x=trig["th"], y=trig["Model_Score"], mode="markers", name="Sustained trigger",
                               marker=dict(color="#EF4444", size=4)), row=2, col=1)
dark(fig, 540, 2)
stretch(st.plotly_chart, fig)

force = st.checkbox("🔓 Show full diagnostic suite even if nominal", value=False)
if is_anom or force:
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Ionospheric dynamics", "📡 RF & scintillation proxy",
                                      "🧭 Navigation & geometry", "🌌 Sky plot & export"])
    with tab1:
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=("sTEC vs thin-shell VTEC [TECU]", "ROT [TECU/min]", "ROTI [TECU/min] (5-min std of ROT)"))
        add_line(f, dp, "sTEC", "sTEC", "#38BDF8", 1)
        if not add_line(f, dp, "VTEC", f"VTEC ({h_ion} km)", "#A855F7", 1, 1.4):
            st.info("VTEC unavailable: no elevation column.")
        add_line(f, dp, "ROT", "ROT", "#FBBF24", 2, 1)
        add_line(f, dp, "ROTI", "ROTI", "#F43F5E", 3, 1.3)
        f.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="0.5 TECU/min", row=3, col=1)
        dark(f, 720, 3)
        stretch(st.plotly_chart, f)
        st.caption("Code-derived TEC is noisy: ROTI has a noise floor set by pseudorange noise. Carrier-phase levelling would lower it.")
    with tab2:
        f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                          subplot_titles=("C/N0 [dB-Hz]", "S4 proxy (60 s window of linear C/N0, uncorrected)"))
        a1 = add_line(f, dp, "CN0_L5", "L5 C/N0", "#10B981", 1)
        a2 = add_line(f, dp, "CN0_S", "S C/N0", "#6366F1", 1)
        add_line(f, dp, "S4_L5", "L5 S4 proxy", "#EF4444", 2, 1.2)
        add_line(f, dp, "S4_S", "S S4 proxy", "#14B8A6", 2, 1.0)
        f.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="S4 = 0.3", row=2, col=1)
        dark(f, 600, 2)
        if a1 or a2:
            stretch(st.plotly_chart, f)
        else:
            st.info("No C/N0 columns mapped for this channel.")
    with tab3:
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=("DOP (from real SV El/Az)", "Position scatter / error [m]", "Locked SVs (lock ≥ threshold, above mask)"))
        got = [add_line(f, dp, k, k, c, 1, 1.2) for k, c in (("GDOP", "#94A3B8"), ("PDOP", "#38BDF8"), ("HDOP", "#10B981"), ("VDOP", "#F59E0B"))]
        add_line(f, dp, "Pos_Err_3D", "3D", "#F43F5E", 2, 1.2)
        add_line(f, dp, "Pos_Err_H", "Horizontal", "#FB923C", 2, 1.0)
        if "Num_SV" in dp and dp["Num_SV"].notna().any():
            f.add_trace(go.Scattergl(x=dp["th"], y=dp["Num_SV"], name="Locked SVs", line=dict(color="#22C55E", width=1.5),
                                     fill="tozeroy"), row=3, col=1)
        dark(f, 720, 3)
        stretch(st.plotly_chart, f)
        if not any(got):
            st.info("DOP unavailable: need El and Az columns for ≥ 4 channels in the L5 file.")
    with tab4:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Sky plot (selected PRN)")
            sky = view.dropna(subset=["El", "Az", "sTEC"])
            if len(sky):
                sky = sky.iloc[:: max(1, len(sky) // 3000)]
                fs = go.Figure(go.Scatterpolar(r=90 - sky["El"], theta=sky["Az"], mode="markers+lines",
                               marker=dict(size=6, color=sky["sTEC"], colorscale="Turbo", showscale=True,
                                           colorbar=dict(title="sTEC")),
                               line=dict(color="rgba(148,163,184,.4)", width=1), name=f"PRN {ch}"))
                fs.update_layout(template="plotly_dark", height=460,
                                 polar=dict(radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                                                            ticktext=["90° zenith", "60°", "30°", "0° horizon"]),
                                            angularaxis=dict(direction="clockwise", rotation=90)))
                stretch(st.plotly_chart, fs)
            else:
                st.info("No elevation/azimuth columns for this channel - sky plot not available.")
        with c2:
            st.subheader("Inspector & export")
            cols = ["t", "sTEC", "VTEC", "ROT", "ROTI", "CN0_L5", "S4_L5", "PDOP", "Pos_Err_3D", "Num_SV", "Model_Score", "Trigger"]
            stretch(st.dataframe, view[cols].iloc[:: max(1, len(view) // 1000)], height=300)
            if st.checkbox("Prepare CSV of selected range"):
                st.download_button("📥 Download processed CSV", view.to_csv(index=False).encode(),
                                   f"NavIC_PRN{ch}_processed.csv", "text/csv")
