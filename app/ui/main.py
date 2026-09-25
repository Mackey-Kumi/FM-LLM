"""
Streamlit UI for GridForecast - FM-LLM forecasting of transformer load & oil temperature
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="GridForecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"
HORIZONS = (96, 192, 336, 720)
COLORS = px.colors.qualitative.Plotly

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.25rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 0 24px; border-radius: 4px 4px 0 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(path: str, **params):
    response = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, timeout: int = 300, **kwargs) -> Optional[dict]:
    try:
        response = requests.post(f"{API_BASE}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API error: {detail}")
        return None
    return response.json()


def fetch_health():
    # Not cached: a failure while the API is still starting must not stick.
    try:
        return api_get("/health")
    except requests.RequestException:
        return None


@st.cache_data(ttl=300)
def fetch_checkpoints():
    return api_get("/checkpoints")["checkpoints"]


@st.cache_data(ttl=300)
def fetch_channels():
    return api_get("/channels")


@st.cache_data(ttl=300)
def fetch_windows(checkpoint: str):
    return api_get("/windows", checkpoint=checkpoint)


@st.cache_data(ttl=300)
def fetch_results():
    return api_get("/results")


class APIError(Exception):
    pass


@st.cache_data(show_spinner=False)
def _fetch_forecast_cached(window_idx: int, horizon: int, checkpoint: str):
    # Raises instead of returning None so failures are never cached.
    response = requests.post(
        f"{API_BASE}/forecast",
        json={"window_idx": window_idx, "horizon": horizon, "checkpoint": checkpoint},
        timeout=300,
    )
    if response.status_code != 200:
        raise APIError(response.json().get("detail", response.text))
    return response.json()


def fetch_forecast(window_idx: int, horizon: int, checkpoint: str) -> Optional[dict]:
    try:
        return _fetch_forecast_cached(window_idx, horizon, checkpoint)
    except (APIError, requests.RequestException) as e:
        st.error(f"Forecast failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def column(rows: List[List[float]], i: int) -> np.ndarray:
    return np.asarray(rows)[:, i]


def plot_forecast(result: Dict, selected_channels: List[str]):
    fig = go.Figure()
    for i, ch in enumerate(result["channels"]):
        if ch not in selected_channels:
            continue
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=result["history_dates"], y=column(result["history"], i),
            mode="lines", name=f"{ch} history", line=dict(color=color, width=1.5),
            opacity=0.45, legendgroup=ch,
        ))
        fig.add_trace(go.Scatter(
            x=result["dates"], y=column(result["ground_truth"], i),
            mode="lines", name=f"{ch} actual", line=dict(color=color, width=1.5),
            legendgroup=ch,
        ))
        fig.add_trace(go.Scatter(
            x=result["dates"], y=column(result["predictions"], i),
            mode="lines", name=f"{ch} forecast", line=dict(color=color, width=2.5, dash="dash"),
            legendgroup=ch,
        ))
    fig.add_vline(x=result["dates"][0], line_dash="dot", line_color="gray")
    fig.add_annotation(x=result["dates"][0], y=1, yref="paper", text="forecast starts",
                       showarrow=False, xanchor="left", yanchor="bottom")
    fig.update_layout(
        title=f"Forecast vs actual — next {result['horizon']}h ({result['horizon'] // 24} days)",
        xaxis_title="Date", yaxis_title="Value (real units)",
        hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_backtest_results(results: Dict):
    horizons = sorted(int(h) for h in results)
    fig = go.Figure()
    for metric in ("mse", "mae"):
        fig.add_trace(go.Scatter(
            x=horizons,
            y=[results[str(h)][f"{metric}_mean"] for h in horizons],
            error_y=dict(type="data", array=[results[str(h)][f"{metric}_std"] for h in horizons]),
            mode="lines+markers", name=metric.upper(),
        ))
    fig.update_layout(
        title="Error vs horizon (mean ± std across windows, normalized units)",
        xaxis_title="Horizon (hours)", yaxis_title="Error", height=400,
    )
    return fig


def plot_scenario(result: Dict, selected_channels: List[str]):
    fig = go.Figure()
    for i, ch in enumerate(result["channels"]):
        if ch not in selected_channels:
            continue
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=result["dates"], y=column(result["base"], i),
                                 mode="lines", name=f"{ch} base", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=result["dates"], y=column(result["predictions"], i),
                                 mode="lines", name=f"{ch} scenario",
                                 line=dict(color=color, width=2, dash="dot")))
    fig.update_layout(
        title="Scenario: base vs perturbed forecast",
        xaxis_title="Date", yaxis_title="Value (real units)",
        hovermode="x unified", height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
health = fetch_health()

with st.sidebar:
    st.markdown('<div class="main-header">⚡ GridForecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">FM-LLM transformer load & oil-temperature forecasting</div>',
                unsafe_allow_html=True)
    st.divider()

    if not health:
        st.error("❌ API disconnected")
        st.caption("Start it with: `python -m app.main` (or `python -m app.api.main`)")

if not health:
    st.markdown('<div class="main-header">⚡ GridForecast</div>', unsafe_allow_html=True)
    st.error("⚠️ Backend API not running (or still loading). Start it with `python -m app.main`, then refresh.")
    st.stop()

checkpoints = fetch_checkpoints()
if not checkpoints:
    st.error("No model checkpoints or precomputed forecasts found. See app/README.md → Setup.")
    st.stop()

channel_info = fetch_channels()
channels = channel_info["channels"]
channel_desc = channel_info["descriptions"]
ckpt_by_name = {c["name"]: c for c in checkpoints}

with st.sidebar:
    mode = "live" if health["live_checkpoints"] else "precomputed"
    st.success(f"✅ API connected ({mode} mode)")
    st.divider()

    selected_checkpoint = st.selectbox(
        "Model checkpoint",
        list(ckpt_by_name),
        format_func=lambda n: f"{n} — {ckpt_by_name[n]['description']}",
    )
    ckpt = ckpt_by_name[selected_checkpoint]
    if ckpt["live"]:
        st.caption("Live inference available (Llama-3.2-1B).")
    else:
        st.caption(f"Precomputed forecasts: {ckpt['precomputed_windows']} windows.")

    window_info = fetch_windows(selected_checkpoint)
    windows = window_info["windows"]
    start_dates = dict(zip(windows, window_info["start_dates"]))
    precomputed_windows = set(window_info["precomputed"])
    window_idx = st.selectbox(
        "Forecast start (test window)",
        windows,
        index=0,
        format_func=lambda w: f"{start_dates[w][:16]}  (#{w}){'' if w in precomputed_windows or not precomputed_windows else ' · live'}",
        help="Each window uses the previous 672h (28 days) as input.",
    )

    default_channels = [c for c in ("OT", "HUFL") if c in channels]
    selected_channels = st.multiselect(
        "Channels to display", channels, default=default_channels,
        format_func=lambda c: f"{c} — {channel_desc.get(c, c)}",
    )

    st.divider()
    st.markdown("### Quick links")
    st.markdown(f"- [API docs]({API_BASE}/docs)")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ GridForecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">28 days of hourly history → up to 30 days of hourly forecasts '
    'for a power transformer\'s load and oil temperature (ETTh1)</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Forecast",
    "🔮 Scenario Simulator",
    "📈 Backtest & Metrics",
    "🚨 Alerts",
    "🔬 Model Explorer",
])

# Tab 1: Forecast
with tab1:
    horizon = st.radio(
        "Forecast horizon", HORIZONS, index=1, horizontal=True,
        format_func=lambda h: f"{h}h ({h // 24}d)",
    )
    with st.spinner(f"Generating {horizon}h forecast..."):
        result = fetch_forecast(window_idx, horizon, selected_checkpoint)

    if result:
        st.session_state["last_forecast"] = result
        ot_i = result["channels"].index("OT")
        ot_mae = float(np.abs(column(result["predictions"], ot_i) - column(result["ground_truth"], ot_i)).mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE (normalized)", f"{result['metrics']['mse']:.3f}",
                  help="Same scale as the paper's Table A.12")
        c2.metric("MAE (normalized)", f"{result['metrics']['mae']:.3f}")
        c3.metric("Oil temp. MAE", f"{ot_mae:.2f} °C", help="Average absolute error in real units")
        c4.metric("Source", result["source"])

        st.plotly_chart(plot_forecast(result, selected_channels), use_container_width=True)

        with st.expander("📋 Per-channel error & raw data"):
            st.dataframe(
                pd.DataFrame(result["per_channel_metrics"]).T.rename(columns=str.upper)
                .style.format("{:.4f}"),
                use_container_width=True,
            )
            df = pd.concat([
                pd.DataFrame(result["predictions"], columns=result["channels"]).add_suffix(" (forecast)"),
                pd.DataFrame(result["ground_truth"], columns=result["channels"]).add_suffix(" (actual)"),
            ], axis=1)
            df.insert(0, "date", result["dates"])
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "📥 Download CSV", df.to_csv(index=False),
                f"forecast_{selected_checkpoint}_w{window_idx}_h{horizon}.csv", "text/csv",
            )

# Tab 2: Scenario Simulator
with tab2:
    st.header("Scenario Simulator")
    st.markdown(
        "What if the load over the **last 4 days** (96h) had been different? "
        "Scale channels up or down and compare the model's forecast to the base case."
    )

    if not ckpt["live"]:
        st.info(
            "Scenarios run the model live, which needs the checkpoint files and Llama-3.2-1B "
            "access (HF_TOKEN). This checkpoint is serving precomputed forecasts only."
        )
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            scen_horizon = st.selectbox("Horizon", HORIZONS, index=1, key="scen_horizon",
                                        format_func=lambda h: f"{h}h ({h // 24}d)")
            perturbations = {}
            for ch in channels:
                pct = st.slider(f"{ch} — {channel_desc.get(ch, ch)}", -50, 50, 0, step=5,
                                format="%d%%", key=f"pert_{ch}")
                if pct:
                    perturbations[ch] = pct / 100

            if st.button("🔮 Run scenario", type="primary", use_container_width=True):
                if not perturbations:
                    st.warning("Move at least one slider.")
                else:
                    with st.spinner("Running base and scenario forecasts live..."):
                        scen = api_post("/scenario", json={
                            "window_idx": window_idx, "horizon": scen_horizon,
                            "checkpoint": selected_checkpoint, "perturbations": perturbations,
                        })
                    if scen:
                        st.session_state["scenario_result"] = scen

        with col2:
            scen = st.session_state.get("scenario_result")
            if scen:
                st.plotly_chart(plot_scenario(scen, selected_channels), use_container_width=True)
                st.subheader("Impact summary (real units)")
                rows = []
                for i, ch in enumerate(scen["channels"]):
                    diff = column(scen["predictions"], i) - column(scen["base"], i)
                    rows.append({"Channel": ch, "Mean change": diff.mean(),
                                 "Max change": diff.max(), "Min change": diff.min()})
                st.dataframe(pd.DataFrame(rows).style.format(precision=3, subset=["Mean change", "Max change", "Min change"]),
                             use_container_width=True)

# Tab 3: Backtest & Metrics
with tab3:
    st.header("Backtest & Model Metrics")
    col1, col2 = st.columns([1, 3])

    with col1:
        max_bt = ckpt["precomputed_windows"] or 200
        bt_windows = st.slider("Number of windows", 1, min(200, max_bt), min(50, max_bt))
        bt_horizons = st.multiselect("Horizons", HORIZONS, default=list(HORIZONS))
        if not ckpt["precomputed_windows"]:
            st.caption("⚠️ No precomputed forecasts: every window runs live (~10s+ each on CPU).")
        if st.button("📊 Run backtest", type="primary", use_container_width=True) and bt_horizons:
            with st.spinner(f"Backtesting {bt_windows} windows..."):
                bt = api_post("/backtest", timeout=3600, json={
                    "checkpoint": selected_checkpoint, "num_windows": bt_windows,
                    "horizons": sorted(bt_horizons),
                })
            if bt:
                st.session_state["backtest"] = {**bt, "checkpoint": selected_checkpoint}

    with col2:
        bt = st.session_state.get("backtest")
        if bt:
            res = bt["results"]
            st.subheader(f"Backtest: {bt['checkpoint']}, {len(bt['windows'])} windows")
            st.dataframe(pd.DataFrame([
                {"Horizon": f"{h}h", "MSE": r["mse_mean"], "MSE std": r["mse_std"],
                 "MAE": r["mae_mean"], "MAE std": r["mae_std"]}
                for h, r in sorted(res.items(), key=lambda kv: int(kv[0]))
            ]).style.format(precision=4, subset=["MSE", "MSE std", "MAE", "MAE std"]),
                use_container_width=True)
            st.plotly_chart(plot_backtest_results(res), use_container_width=True)

        st.subheader("Full test set vs paper (Table A.12)")
        reported = fetch_results()
        paper = reported["paper"]
        ours = reported["checkpoints"].get(selected_checkpoint, {})
        rows = []
        for h in ("192", "336", "720"):
            if h in ours:
                rows.append({
                    "Horizon": f"{h}h",
                    "Our MSE": ours[h][0], "Paper MSE": paper[h][0],
                    "Our MAE": ours[h][1], "Paper MAE": paper[h][1],
                    "MSE gap": f"{(ours[h][0] / paper[h][0] - 1) * 100:+.0f}%",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("All 2161 test windows, from kaggle/experiments/10_rollout_eval_instnorm. "
                   "The backtest above uses a subset of windows, so its numbers differ slightly.")

# Tab 4: Alerts
with tab4:
    st.header("Overheating & load alerts")
    st.markdown("Flag forecast hours where a channel is predicted to leave its safe band, "
                "and check those warnings against what actually happened.")

    result = st.session_state.get("last_forecast")
    if not result:
        st.info("Generate a forecast in the first tab.")
    else:
        col1, col2 = st.columns([1, 2])
        ch_idx_default = channels.index("OT") if "OT" in channels else 0
        with col1:
            alert_channel = st.selectbox("Monitor channel", channels, index=ch_idx_default,
                                         format_func=lambda c: f"{c} — {channel_desc.get(c, c)}")
            i = result["channels"].index(alert_channel)
            hist = column(result["history"], i)
            pred = column(result["predictions"], i)
            truth = column(result["ground_truth"], i)
            upper = st.number_input("Upper limit", value=float(np.round(np.percentile(hist, 95), 1)), step=0.5,
                                    help="Default: 95th percentile of the past week")
            lower = st.number_input("Lower limit", value=float(np.round(np.percentile(hist, 5), 1)), step=0.5,
                                    help="Default: 5th percentile of the past week")

            pred_bad = (pred > upper) | (pred < lower)
            true_bad = (truth > upper) | (truth < lower)
            st.divider()
            if pred_bad.any():
                first = int(np.argmax(pred_bad))
                st.error(f"⚠️ {int(pred_bad.sum())} forecast hours outside limits. "
                         f"First breach in **{first}h** ({result['dates'][first][:16]}).")
            else:
                st.success("✅ No breaches forecast in this horizon.")

            hits = int((pred_bad & true_bad).sum())
            st.caption(
                f"Check against actuals: {int(true_bad.sum())} hours actually outside limits; "
                f"{hits} of them were forecast; {int((pred_bad & ~true_bad).sum())} false alarms."
            )

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result["history_dates"], y=hist, name="History",
                                     line=dict(color="gray", width=1)))
            fig.add_trace(go.Scatter(x=result["dates"], y=truth, name="Actual",
                                     line=dict(color="black", width=1.5)))
            fig.add_trace(go.Scatter(x=result["dates"], y=pred, name="Forecast",
                                     line=dict(color="#1f77b4", width=2.5, dash="dash")))
            breach_x = [d for d, b in zip(result["dates"], pred_bad) if b]
            breach_y = pred[pred_bad]
            fig.add_trace(go.Scatter(x=breach_x, y=breach_y, mode="markers", name="Forecast breach",
                                     marker=dict(color="red", size=6)))
            fig.add_hline(y=upper, line_dash="dash", line_color="red", annotation_text="Upper")
            fig.add_hline(y=lower, line_dash="dash", line_color="red", annotation_text="Lower")
            fig.update_layout(title=f"{alert_channel} forecast with alert limits", height=450,
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# Tab 5: Model Explorer
with tab5:
    st.header("Model Explorer")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Architecture")
        st.markdown("""
        **FM-LLM: Frequency-Enhanced Mixture-of-Experts for Time Series Forecasting**

        1. **Fourier Embedding Module** (Eq. 7-11)
           - Patch length 96 → MLP(512) → FAN → Linear(2048)
           - Fourier Analysis Network splits periodic and trend components
        2. **Frozen Llama-3.2-1B backbone**
           - 7 tokens (672h = 28 days) per channel, weights frozen
        3. **FAN-MoE decoder** (Eq. 13-18)
           - 2 shared FAN experts + 2 routed experts, top-1 gate
           - Load-balanced routing with bias correction
        4. **Autoregressive rollout**
           - Sliding 7-token window, 8 steps → 768h, truncated to the horizon
        """)
        st.subheader("Checkpoints")
        for c in checkpoints:
            st.markdown(f"- **{c['name']}**: {c['description']}")

    with col2:
        st.subheader("Checkpoint comparison (full ETTh1 test set)")
        reported = fetch_results()
        rows = [
            {"Horizon": f"{h}h", "Checkpoint": name, "MSE": v[0], "MAE": v[1]}
            for name, by_h in reported["checkpoints"].items()
            for h, v in by_h.items()
        ]
        df_comp = pd.DataFrame(rows)
        for metric in ("MSE", "MAE"):
            st.markdown(f"**{metric}** (lower is better)")
            st.dataframe(
                df_comp.pivot(index="Horizon", columns="Checkpoint", values=metric)
                .style.highlight_min(axis=1, color="lightgreen").format("{:.4f}"),
                use_container_width=True,
            )
        st.info("""
        **Key finding:** per-window instance normalization cut 720h MSE from 0.6381 to 0.4721
        (−26%), shrinking the gap to the paper from ~61% to ~19%.
        """)

st.divider()
st.caption("GridForecast v0.2.0 | FM-LLM reproduction | Streamlit + FastAPI")
