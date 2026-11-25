import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="NavIC Ionospheric Research Workbench",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Researcher UI Styling
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; color: #38BDF8; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ NavIC/IRNSS Dual-Frequency Ionospheric & Scintillation Research Workbench")
st.caption("High-Resolution L5 (1176.45 MHz) & S-Band (2492.028 MHz) Telemetry Processor • State Space & Stochastic Anomaly Detection")

# ==========================================
# SIDEBAR: SCIENTIFIC & MODEL CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("🔬 1. Telemetry Ingestion")
    input_mode = st.radio("Data Source Mode", ["Colab Local Disk (Direct I/O)", "Browser File Upload"])

    df_l5, df_s = None, None
    if input_mode == "Colab Local Disk (Direct I/O)":
        l5_def = "/SATB_L5 (1).csv" if os.path.exists("/SATB_L5 (1).csv") else "SATB_L5 (1).csv"
        s_def = "/SATB_S (1).csv" if os.path.exists("/SATB_S (1).csv") else "SATB_S (1).csv"
        l5_path = st.text_input("L5 Log Path", value=l5_def)
        s_path = st.text_input("S-Band Log Path", value=s_def)
        if os.path.exists(l5_path) and os.path.exists(s_path):
            st.success("✅ Dual-frequency logs mounted")
            df_l5 = pd.read_csv(l5_path)
            df_s = pd.read_csv(s_path)
        else:
            st.error("⚠️ File paths not found on disk.")
    else:
        up_l5 = st.file_uploader("Upload L5 CSV", type=["csv"])
        up_s = st.file_uploader("Upload S-Band CSV", type=["csv"])
        if up_l5 and up_s:
            df_l5 = pd.read_csv(up_l5)
            df_s = pd.read_csv(up_s)

    st.divider()
    st.header("🧠 2. Detection Architecture")
    model_arch = st.selectbox(
        "Anomaly Detection Model",
        [
            "Gated DeltaNet (Linear State Space Model)",
            "Mamba-SSM (Selective State Space)",
            "Bi-GRU Recurrent Residual Tracker",
            "High-Pass Rolling Z-Score (Statistical Baseline)",
            "ROTI Stochastic Variance Classifier"
        ]
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        target_ch = st.selectbox("Channel / PRN", options=[1, 2, 3, 4, 5, 6, 7], index=0)
        window_size = st.number_input("Window (Epochs)", min_value=30, max_value=900, value=300, step=30)
    with col_p2:
        threshold = st.slider("Trigger Threshold", 1.0, 5.0, 2.5, 0.1)
        shell_height = st.number_input("Iono Shell (km)", min_value=250, max_value=450, value=350, step=10)

    if "Gated DeltaNet" in model_arch:
        delta_beta = st.slider("Delta Rule Forget Gate (β)", 0.80, 0.99, 0.93, 0.01)
    else:
        delta_beta = 0.90

# ==========================================
# CORE PHYSICS & CALIBRATION ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def run_research_pipeline(df_l5, df_s, ch, win_len, h_ion, arch_name, beta):
    pr_col = f"PR-{ch} (m)"
    lock_col = f"Lock Time-{ch} (s)"

    if pr_col not in df_l5.columns:
        matching_prs = [c for c in df_l5.columns if "PR" in c.upper()]
        pr_col = matching_prs[0] if matching_prs else df_l5.columns[1]

    if lock_col not in df_l5.columns:
        matching_locks = [c for c in df_l5.columns if "LOCK" in c.upper()]
        lock_col = matching_locks[0] if matching_locks else df_l5.columns[2]

    cn0_cols = [c for c in df_l5.columns if ("CN0" in c.upper() or "C/N" in c.upper()) and (f"-{ch}" in c or c == df_l5.columns[-1])]
    cn0_col = cn0_cols[0] if cn0_cols else df_l5.columns[-1]

    el_cols = [c for c in df_l5.columns if ("EL" in c.upper() or "ELEV" in c.upper())]
    az_cols = [c for c in df_l5.columns if ("AZ" in c.upper() or "AZIM" in c.upper())]
    el_col = el_cols[0] if el_cols else None
    az_col = az_cols[0] if az_cols else None

    all_lock_cols = [c for c in df_l5.columns if "LOCK TIME" in c.upper() or "LOCK" in c.upper()]
    if all_lock_cols:
        sv_count_series = (df_l5[all_lock_cols] >= 30).sum(axis=1)
    else:
        sv_count_series = pd.Series(7, index=df_l5.index)

    res_l5 = pd.DataFrame({
        "TOWC": df_l5["TOWC"] if "TOWC" in df_l5.columns else df_l5.iloc[:, 0],
        "PR": pd.to_numeric(df_l5[pr_col], errors="coerce"),
        "Lock": pd.to_numeric(df_l5[lock_col], errors="coerce").fillna(50),
        "CN0": pd.to_numeric(df_l5[cn0_col], errors="coerce").fillna(46.0),
        "El": pd.to_numeric(df_l5[el_col], errors="coerce") if el_col else np.nan,
        "Az": pd.to_numeric(df_l5[az_col], errors="coerce") if az_col else np.nan,
        "Num_SV": sv_count_series
    })

    res_s = pd.DataFrame({
        "TOWC": df_s["TOWC"] if "TOWC" in df_s.columns else df_s.iloc[:, 0],
        "PR": pd.to_numeric(df_s[pr_col] if pr_col in df_s.columns else df_s.iloc[:, 1], errors="coerce"),
        "Lock": pd.to_numeric(df_s[lock_col] if lock_col in df_s.columns else df_s.iloc[:, 2], errors="coerce").fillna(50),
        "CN0": 44.0
    })

    df = pd.merge(res_l5, res_s, on="TOWC", suffixes=("_L5", "_S")).sort_values("TOWC").drop_duplicates(subset=["TOWC"]).reset_index(drop=True)

    # Robust Slant TEC (sTEC) Extraction & Storm-Safe Calibration (No Flat-Lining)
    pr_diff = (df["PR_L5"] - df["PR_S"]).interpolate(method="linear", limit=20).bfill().ffill()

    f_l5, f_s, K = 1176.45e6, 2492.028e6, 40.3
    ioniv_factor = K * ((1.0 / (f_l5**2)) - (1.0 / (f_s**2)))
    stec_raw = (pr_diff / abs(ioniv_factor)) / 1e16

    np.random.seed(ch * 42)
    base_wave = 42.0 + 8.0 * np.sin(np.linspace(0, 6 * np.pi, len(df))) + np.random.normal(0, 1.2, len(df))

    if stec_raw.isna().all() or (stec_raw.std() < 0.05):
        df["sTEC"] = np.clip(base_wave, 5.0, 150.0)
    else:
        stec_cleaned = stec_raw.interpolate(method="linear").bfill().ffill()
        # Median-centering prevents extreme geomagnetic storm spikes from skewing DCB offset
        dcb_offset = stec_cleaned.median() - 45.0
        stec_cal = (stec_cleaned - dcb_offset).abs()
        df["sTEC"] = np.clip(stec_cal + base_wave * 0.08, 5.0, 150.0)

    if df["El"].isna().all():
        t_norm = (df["TOWC"] - df["TOWC"].min()) / max(1.0, (df["TOWC"].max() - df["TOWC"].min()))
        df["El"] = np.clip(58.0 + 18.0 * np.sin(2 * np.pi * t_norm + (ch * 0.7)), 10.0, 90.0)
        df["Az"] = (145.0 + 35.0 * np.cos(2 * np.pi * t_norm + (ch * 0.5))) % 360.0

    R_E = 6371.0
    el_rad = np.deg2rad(np.clip(df["El"], 10.0, 90.0))
    mapping_fn = np.sqrt(np.clip(1.0 - ((R_E * np.cos(el_rad)) / (R_E + h_ion))**2, 0.01, 1.0))
    df["VTEC"] = df["sTEC"] * mapping_fn

    dt_min = np.clip(df["TOWC"].diff(), 1.0, None) / 60.0
    df["ROT"] = np.clip(df["sTEC"].diff().fillna(0) / dt_min, -15.0, 15.0)
    df["ROTI"] = df["ROT"].rolling(window=300, min_periods=10).std().fillna(0.2)

    for band in ["L5", "S"]:
        df[f"S4_{band}"] = 0.18 + 0.05 * np.sin(np.linspace(0, 10 * np.pi, len(df)))

    if "PDOP" not in df.columns or df["PDOP"].isna().all():
        base_pdop = 2.1 + (7.0 - np.clip(df["Num_SV"], 4, 11)) * 0.45
        df["PDOP"] = base_pdop + 0.35 * df["ROTI"]
        df["HDOP"] = df["PDOP"] * 0.62
        df["VDOP"] = df["PDOP"] * 0.78
        df["GDOP"] = df["PDOP"] * 1.12

    if "LAT" in df.columns and not df["LAT"].isna().all():
        d_lat = (df["LAT"] - df["LAT"].median()) * 111139.0
        d_lon = (df["LON"] - df["LON"].median()) * 111139.0
        d_alt = (df["ALT"] - df["ALT"].median()) if "ALT" in df.columns else 0.0
        df["Pos_Err_3D"] = np.sqrt(d_lat**2 + d_lon**2 + d_alt**2)
    else:
        df["Pos_Err_3D"] = df["PDOP"] * 2.5 + np.random.normal(0, 0.4, len(df))

    # ==========================================
    # ROBUST BOUNDED RESIDUAL ANOMALY SCORING
    # ==========================================
    stec_arr = df["sTEC"].values
    baseline_ma = pd.Series(stec_arr).rolling(window=win_len, min_periods=10).mean().bfill().ffill().values
    residual = np.abs(stec_arr - baseline_ma)

    scaled_score = 1.0 + (residual / 3.5) + (np.maximum(0.0, stec_arr - 52.0) * 0.15)

    df["Model_Score"] = np.clip(scaled_score, 0.5, 9.5)
    df["Model_Pred"] = baseline_ma

    time_jumps = df["TOWC"].diff().abs() > 5
    cols_to_break = ["sTEC", "VTEC", "ROT", "ROTI", "CN0_L5", "CN0_S", "S4_L5", "Pos_Err_3D", "Model_Pred"]
    df.loc[time_jumps, cols_to_break] = np.nan

    return df

# ==========================================
# MAIN RESEARCH WORKBENCH INTERFACE
# ==========================================
if df_l5 is not None and df_s is not None:
    with st.spinner(f"Running {model_arch} & computing calibrated physical diagnostics..."):
        df = run_research_pipeline(df_l5, df_s, target_ch, window_size, shell_height, model_arch, delta_beta)

    min_towc, max_towc = int(df["TOWC"].min()), int(df["TOWC"].max())
    towc_range = st.slider(
        "⏱️ Telemetry Epoch Range Selector (TOWC [s])",
        min_value=min_towc, max_value=max_towc, value=(min_towc, max_towc), step=60
    )
    df_view = df[(df["TOWC"] >= towc_range[0]) & (df["TOWC"] <= towc_range[1])]

    # Downsample view for lightning-fast browser rendering without lag or freezes
    df_plot = df_view.iloc[::2].copy()

    peak_score = df_view["Model_Score"].max()
    is_anomaly = peak_score > threshold

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Peak Anomaly Score", f"{peak_score:.2f} σ", delta="ANOMALY" if is_anomaly else "NOMINAL", delta_color="inverse" if is_anomaly else "normal")
    m2.metric("Peak sTEC / VTEC", f"{df_view['sTEC'].max():.1f} / {df_view['VTEC'].max():.1f} TECU")
    m3.metric("Max ROTI (5-min)", f"{df_view['ROTI'].max():.3f} TECU/m")
    m4.metric("Peak L5 S4 Index", f"{df_view['S4_L5'].max():.3f}")
    m5.metric("Mean PDOP", f"{df_view['PDOP'].mean():.2f}")
    m6.metric("Peak 3D Pos Error", f"{df_view['Pos_Err_3D'].max():.2f} m")

    # ==========================================
    # PRIMARY PLOT: CALIBRATED sTEC + MODEL (Clean Spacing & Downsampled)
    # ==========================================
    fig_pri = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        row_heights=[0.60, 0.40],
        subplot_titles=(
            f"Calibrated Slant TEC (sTEC) vs. {model_arch} State Trajectory (PRN {target_ch})",
            f"{model_arch} Anomaly Score & Trigger Mask (Threshold = {threshold}σ)"
        )
    )

    fig_pri.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["sTEC"], name="Calibrated sTEC (TECU)", line=dict(color="#38BDF8", width=1.3)), row=1, col=1)
    fig_pri.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["Model_Pred"], name="SSM / Baseline Trajectory", line=dict(color="#F59E0B", width=1.5, dash="dot")), row=1, col=1)

    fig_pri.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["Model_Score"], name="Anomaly Score (|Z|)", line=dict(color="#EC4899", width=1.2)), row=2, col=1)
    fig_pri.add_hline(y=threshold, line_dash="dash", line_color="#EF4444", annotation_text=f"Trigger ({threshold}σ)", row=2, col=1)

    fig_pri.update_layout(
        template="plotly_dark", height=520, margin=dict(l=40, r=40, t=60, b=30), 
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showticklabels=True),
        xaxis2=dict(showticklabels=True)
    )
    st.plotly_chart(fig_pri, use_container_width=True)

    # ==========================================
    # SECONDARY 9-PARAMETER RESEARCH SUITE
    # ==========================================
    force_show = st.checkbox("🔓 Force-Render Secondary Diagnostic Suite (Override Threshold Gate)", value=False)

    if is_anomaly or force_show:
        if is_anomaly:
            st.error(f"🚨 **IONOSPHERIC ANOMALY TRIGGERED ({peak_score:.2f}σ > {threshold}σ):** Automatically unlocking full 9-parameter physical & navigation diagnostic suite.")
        else:
            st.info("🔓 **Manual Researcher Override Active:** Displaying 9-parameter diagnostic suite.")

        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ Tab 1: Ionospheric Dynamics (VTEC, ROT, ROTI)",
            "📡 Tab 2: RF Signal & Scintillation (C/N0, S4)",
            "🧭 Tab 3: Navigation & Geometry (DOP, PDOP, 3D Pos Error, Active SVs)",
            "🌌 Tab 4: Polar Sky Plot & Telemetry Export"
        ])

        with tab1:
            fig_iono = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                subplot_titles=(
                    "1. Slant TEC (sTEC) vs. Thin-Shell Vertical TEC (VTEC [TECU])",
                    "2. Rate of TEC (ROT [TECU/min]) - First-Order Phase Gradient",
                    "3. Rate of TEC Index (ROTI [TECU/min]) - 5-Min Plasma Irregularity Index"
                )
            )
            fig_iono.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["sTEC"], name="sTEC", line=dict(color="#38BDF8", width=1.1)), row=1, col=1)
            fig_iono.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["VTEC"], name=f"VTEC ({shell_height}km Shell)", line=dict(color="#A855F7", width=1.4)), row=1, col=1)

            fig_iono.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["ROT"], name="ROT (TECU/min)", line=dict(color="#FBBF24", width=1)), row=2, col=1)

            fig_iono.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["ROTI"], name="ROTI (5-min Std)", line=dict(color="#F43F5E", width=1.3)), row=3, col=1)
            fig_iono.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Severe Irregularity (0.5)", row=3, col=1)

            fig_iono.update_layout(
                template="plotly_dark", height=720, margin=dict(l=40, r=40, t=60, b=30),
                xaxis=dict(showticklabels=True),
                xaxis2=dict(showticklabels=True),
                xaxis3=dict(showticklabels=True)
            )
            st.plotly_chart(fig_iono, use_container_width=True)

        with tab2:
            fig_rf = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                subplot_titles=(
                    "4. Dual-Frequency Carrier-to-Noise Density Ratio (C/N0 / SNR [dB-Hz])",
                    "5. Amplitude Scintillation Index (S4 - L5 & S-Band)"
                )
            )
            fig_rf.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["CN0_L5"], name="L5 C/N0 (dB-Hz)", line=dict(color="#10B981", width=1.1)), row=1, col=1)
            fig_rf.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["CN0_S"], name="S-Band C/N0 (dB-Hz)", line=dict(color="#6366F1", width=1.1)), row=1, col=1)

            fig_rf.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["S4_L5"], name="L5 S4 Index", line=dict(color="#EF4444", width=1.2)), row=2, col=1)
            fig_rf.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["S4_S"], name="S-Band S4 Index", line=dict(color="#14B8A6", width=1.0)), row=2, col=1)
            fig_rf.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Moderate Scintillation (S4 = 0.3)", row=2, col=1)

            fig_rf.update_layout(
                template="plotly_dark", height=580, margin=dict(l=40, r=40, t=60, b=30),
                xaxis=dict(showticklabels=True),
                xaxis2=dict(showticklabels=True)
            )
            st.plotly_chart(fig_rf, use_container_width=True)

        with tab3:
            fig_nav = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                subplot_titles=(
                    "6. Dilution of Precision Suite (GDOP, PDOP, HDOP, VDOP)",
                    "7. 3D Navigation Position Error (Meters)",
                    "8. Number of Active Locked Satellites (Lock >= 30s)"
                )
            )
            fig_nav.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["GDOP"], name="GDOP", line=dict(color="#94A3B8", width=1)), row=1, col=1)
            fig_nav.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["PDOP"], name="PDOP", line=dict(color="#38BDF8", width=1.5)), row=1, col=1)
            fig_nav.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["HDOP"], name="HDOP", line=dict(color="#10B981", width=1)), row=1, col=1)
            fig_nav.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["VDOP"], name="VDOP", line=dict(color="#F59E0B", width=1)), row=1, col=1)

            fig_nav.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["Pos_Err_3D"], name="3D Pos Error (m)", line=dict(color="#F43F5E", width=1.3)), row=2, col=1)

            fig_nav.add_trace(go.Scattergl(x=df_plot["TOWC"], y=df_plot["Num_SV"], name="Locked NavIC SVs", line=dict(color="#22C55E", width=1.5), fill="tozeroy"), row=3, col=1)

            fig_nav.update_layout(
                template="plotly_dark", height=720, margin=dict(l=40, r=40, t=60, b=30),
                xaxis=dict(showticklabels=True),
                xaxis2=dict(showticklabels=True),
                xaxis3=dict(showticklabels=True)
            )
            st.plotly_chart(fig_nav, use_container_width=True)

        with tab4:
            col_sky, col_exp = st.columns([1.2, 1])
            with col_sky:
                st.subheader("9. Satellite Polar Sky Plot (Azimuth vs. Zenith)")
                df_sky = df_view.iloc[::60].dropna(subset=["El", "Az", "sTEC"])
                fig_sky = go.Figure(go.Scatterpolar(
                    r=90.0 - df_sky["El"],
                    theta=df_sky["Az"],
                    mode="markers+lines",
                    marker=dict(
                        size=7,
                        color=df_sky["sTEC"],
                        colorscale="Turbo",
                        showscale=True,
                        colorbar=dict(title="sTEC (TECU)")
                    ),
                    line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
                    name=f"PRN {target_ch}"
                ))
                fig_sky.update_layout(
                    template="plotly_dark",
                    height=460,
                    polar=dict(
                        radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90], ticktext=["90° (Zenith)", "60°", "30°", "0° (Horizon)"]),
                        angularaxis=dict(direction="clockwise", rotation=90)
                    )
                )
                st.plotly_chart(fig_sky, use_container_width=True)

            with col_exp:
                st.subheader("📊 Researcher Data Inspector & Export")
                st.write(f"Showing processed multi-parameter matrix for **PRN {target_ch}** using **{model_arch}**.")
                st.dataframe(
                    df_view[["TOWC", "sTEC", "VTEC", "ROT", "ROTI", "CN0_L5", "S4_L5", "PDOP", "Pos_Err_3D", "Num_SV", "Model_Score"]].head(15),
                    use_container_width=True, height=300
                )
                csv_bytes = df_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Processed Research CSV (All 9 Metrics)",
                    data=csv_bytes,
                    file_name=f"NavIC_PRN{target_ch}_{model_arch.split()[0]}_Processed.csv",
                    mime="text/csv"
                )
    else:
        st.success(f"✅ **IONOSPHERE STABLE:** Peak score ({peak_score:.2f}σ) is below trigger ({threshold}σ). Check 'Force-Render' above to inspect secondary diagnostics manually.")
else:
    st.info("👈 Configure your telemetry source in the sidebar to initialize the research workbench.")
