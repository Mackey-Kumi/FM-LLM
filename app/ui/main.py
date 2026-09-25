"""
Streamlit UI for GridForecast - transformer overheating early warning with FM-LLM
"""

import json
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
OUTLOOKS = (24, 48, 72, 168)
LOAD_CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")
COLORS = px.colors.qualitative.Plotly
LEVEL_COLORS = {"green": "#2ca02c", "amber": "#ff9f1c", "red": "#d62728"}
UNITS = {"OT": "°C"}

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
class APIError(Exception):
    pass


def api_get(path: str, **params):
    response = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _post_cached(path: str, payload: str):
    # Raises instead of returning None so failures are never cached.
    response = requests.post(f"{API_BASE}{path}", data=payload,
                             headers={"Content-Type": "application/json"}, timeout=3600)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(detail)
    return response.json()


def api_post(path: str, show_error: bool = True, **payload) -> Optional[dict]:
    try:
        return _post_cached(path, json.dumps(payload, sort_keys=True))
    except (APIError, requests.RequestException) as e:
        if show_error:
            st.error(f"{e}")
        return None


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


def fetch_forecast(window_idx: int, horizon: int, checkpoint: str):
    return api_post("/forecast", window_idx=window_idx, horizon=horizon, checkpoint=checkpoint)


# ---------------------------------------------------------------------------
# Formatting & plots
# ---------------------------------------------------------------------------
def column(rows: List[List[float]], i: int) -> np.ndarray:
    return np.asarray(rows)[:, i]


def fmt_hours(h: Optional[float]) -> str:
    if h is None:
        return "—"
    if h < 1:
        return "<1h"
    return f"{h:.0f}h" if h < 48 else f"{h / 24:.1f} days"


def fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.0%}"


def fmt_time(ts: str) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b %H:%M")


def plot_outlook(fc: Dict, ch: str, limit: float, margin: float, outlook_hours: int, reveal: bool):
    i = fc["channels"].index(ch)
    unit = UNITS.get(ch, "")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["history_dates"], y=column(fc["history"], i), name="Past 7 days",
                             line=dict(color="gray", width=1.5)))
    fig.add_trace(go.Scatter(x=fc["dates"], y=column(fc["predictions"], i), name="FM-LLM forecast",
                             line=dict(color="#1f77b4", width=2.5)))
    if reveal:
        fig.add_trace(go.Scatter(x=fc["dates"], y=column(fc["ground_truth"], i), name="What actually happened",
                                 line=dict(color="black", width=1.5, dash="dot")))
    fig.add_vrect(x0=fc["dates"][0], x1=fc["dates"][min(outlook_hours, len(fc["dates"])) - 1],
                  fillcolor="#1f77b4", opacity=0.07, line_width=0,
                  annotation_text=f"{outlook_hours}h outlook", annotation_position="top left")
    if margin > 0:
        fig.add_hrect(y0=limit - margin, y1=limit, fillcolor=LEVEL_COLORS["amber"], opacity=0.12, line_width=0)
    fig.add_hline(y=limit, line_dash="dash", line_color=LEVEL_COLORS["red"],
                  annotation_text=f"Alarm level {limit:g}{unit}", annotation_position="bottom right")
    fig.update_layout(height=430, hovermode="x unified", yaxis_title=f"{ch} ({unit or 'real units'})",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(t=40))
    return fig


def plot_risk_strip(days: List[Dict], limit: float, reveal: bool):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[d["date"] for d in days], y=[d["forecast_peak"] for d in days],
        marker_color=[LEVEL_COLORS[d["level"]] for d in days], name="Forecast daily peak",
    ))
    if reveal:
        fig.add_trace(go.Scatter(x=[d["date"] for d in days], y=[d["actual_peak"] for d in days],
                                 mode="markers", marker=dict(color="black", symbol="diamond", size=7),
                                 name="Actual daily peak"))
    fig.add_hline(y=limit, line_dash="dash", line_color=LEVEL_COLORS["red"])
    fig.update_layout(height=260, title="30-day risk outlook (daily forecast peak)",
                      showlegend=reveal, margin=dict(t=40, b=10))
    return fig


def plot_warning_timeline(ev: Dict):
    labels = ev["labels"]
    fig = go.Figure()
    event_dates = [r["issued_at"] for r in ev["records"] if r["event"]]
    fig.add_trace(go.Scatter(
        x=event_dates, y=["Breach happened"] * len(event_dates), mode="markers",
        marker=dict(color="black", symbol="square", size=9), name="Breach within outlook",
    ))
    styles = {
        "hit": dict(color=LEVEL_COLORS["green"], symbol="circle", size=9),
        "miss": dict(color=LEVEL_COLORS["amber"], symbol="circle-open", size=10, line=dict(width=2)),
        "false alarm": dict(color=LEVEL_COLORS["red"], symbol="x", size=9),
    }
    for kind, marker in styles.items():
        xs, ys = [], []
        for m in ev["methods"]:
            for r in ev["records"]:
                warned = r["warned"][m]
                k = ("hit" if warned else "miss") if r["event"] else ("false alarm" if warned else None)
                if k == kind:
                    xs.append(r["issued_at"])
                    ys.append(labels[m])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=marker, name=kind.capitalize()))
    fig.update_layout(
        height=320, title="Every daily outlook: warnings vs. what happened",
        yaxis=dict(categoryorder="array",
                   categoryarray=[labels[m] for m in reversed(ev["methods"])] + ["Breach happened"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


def plot_forecast(result: Dict, selected_channels: List[str]):
    fig = go.Figure()
    for i, ch in enumerate(result["channels"]):
        if ch not in selected_channels:
            continue
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=result["history_dates"], y=column(result["history"], i), mode="lines",
                                 name=f"{ch} history", line=dict(color=color, width=1.5), opacity=0.45))
        fig.add_trace(go.Scatter(x=result["dates"], y=column(result["ground_truth"], i), mode="lines",
                                 name=f"{ch} actual", line=dict(color=color, width=1.5)))
        fig.add_trace(go.Scatter(x=result["dates"], y=column(result["predictions"], i), mode="lines",
                                 name=f"{ch} forecast", line=dict(color=color, width=2.5, dash="dash")))
    fig.add_vline(x=result["dates"][0], line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Forecast vs actual — next {result['horizon']}h ({result['horizon'] // 24} days)",
        xaxis_title="Date", yaxis_title="Value (real units)", hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_backtest_results(results: Dict):
    horizons = sorted(int(h) for h in results)
    fig = go.Figure()
    for metric in ("mse", "mae"):
        fig.add_trace(go.Scatter(
            x=horizons, y=[results[str(h)][f"{metric}_mean"] for h in horizons],
            error_y=dict(type="data", array=[results[str(h)][f"{metric}_std"] for h in horizons]),
            mode="lines+markers", name=metric.upper(),
        ))
    fig.update_layout(title="Error vs horizon (mean ± std across windows, normalized units)",
                      xaxis_title="Horizon (hours)", yaxis_title="Error", height=380)
    return fig


def first_breach(values: np.ndarray, limit: float) -> Optional[int]:
    above = values >= limit
    return int(np.argmax(above)) if above.any() else None


# ---------------------------------------------------------------------------
# Sidebar: connection + site settings
# ---------------------------------------------------------------------------
health = fetch_health()

with st.sidebar:
    st.markdown('<div class="main-header">⚡ GridForecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transformer overheating early warning</div>', unsafe_allow_html=True)
    if not health:
        st.error("❌ API disconnected")
        st.caption("Start it with: `python -m app.main`")

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
    st.success(f"✅ Connected ({'live' if health['live_checkpoints'] else 'precomputed'} mode)")
    st.subheader("Site settings")
    monitored = st.selectbox("Monitored signal", channels,
                             index=channels.index("OT") if "OT" in channels else 0,
                             format_func=lambda c: f"{c} — {channel_desc.get(c, c)}")
    unit = UNITS.get(monitored, "")
    limit = st.number_input(f"Alarm level ({unit or 'real units'})", value=10.0 if monitored == "OT" else 15.0,
                            step=0.5,
                            help="The test period is winter (Oct–Feb), when oil temperature peaks around "
                                 "15°C; in summer it passes 40°C, so a real site would set this seasonally.")
    margin = st.number_input(f"Early-warning margin ({unit or 'units'})", value=1.0, min_value=0.0, step=0.5,
                             help="Also warn when the forecast comes within this much of the alarm level "
                                  "(forecasts tend to smooth out peaks).")
    outlook_hours = st.select_slider("Warning outlook", OUTLOOKS, value=72,
                                     format_func=lambda h: f"{h}h" if h < 48 else f"{h // 24} days")

    st.subheader("Model")
    selected_checkpoint = st.selectbox(
        "Checkpoint", list(ckpt_by_name),
        format_func=lambda n: f"{n} — {ckpt_by_name[n]['description']}",
    )
    ckpt = ckpt_by_name[selected_checkpoint]
    st.caption("Live inference available." if ckpt["live"]
               else f"Precomputed forecasts: {ckpt['precomputed_windows']} daily outlooks.")

warning_params = dict(checkpoint=selected_checkpoint, channel=monitored, limit=limit,
                      margin=margin, outlook_hours=outlook_hours)
evaluation = api_post("/warning/evaluate", show_error=False, **warning_params)

window_info = fetch_windows(selected_checkpoint)
windows = window_info["windows"]
start_dates = dict(zip(windows, window_info["start_dates"]))

# Open on a day where the model warned correctly at least a day ahead: the most useful demo moment.
default_window = windows[0]
if evaluation:
    caught = [r for r in evaluation["records"] if r["event"] and r["warned"]["fm_llm"]]
    ahead = [r for r in caught if r["actual_first_hour"] >= 24] or caught
    if ahead:
        default_window = ahead[0]["window_idx"]

with st.sidebar:
    window_idx = st.selectbox(
        "Outlook issued at", windows, index=windows.index(default_window),
        format_func=lambda w: pd.Timestamp(start_dates[w]).strftime("%a %d %b %Y %H:%M"),
        help="Each outlook uses the previous 28 days of hourly data.",
    )
    st.divider()
    st.markdown(f"[API docs]({API_BASE}/docs)")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ GridForecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Early warning of transformer overheating: FM-LLM reads 28 days of hourly load '
    'and oil-temperature data and warns operators before the oil temperature crosses its alarm level.</div>',
    unsafe_allow_html=True,
)

tab_warn, tab_perf, tab_whatif, tab_explore, tab_model = st.tabs([
    "🚨 Early Warning",
    "📈 Warning Performance",
    "🔮 Load What-if",
    "📊 Forecast Explorer",
    "🔬 Model & Accuracy",
])

# Tab: Early Warning (operator dashboard)
with tab_warn:
    ol = api_post("/outlook", window_idx=window_idx, **warning_params)
    fc = fetch_forecast(window_idx, 720, selected_checkpoint)
    if ol and fc:
        f = ol["forecast"]
        st.subheader(f"Outlook issued {fmt_time(ol['issued_at'])}")

        if ol["status"] == "in_breach":
            st.warning(f"🟠 **ALREADY ABOVE ALARM LEVEL**: {monitored} is {ol['current']:.1f}{unit} now "
                       f"(alarm level {limit:g}{unit}).")
        elif ol["status"] == "warning":
            when = pd.Timestamp(ol["issued_at"]) + pd.Timedelta(hours=f["first_breach_hour"])
            st.error(f"🔴 **WARNING**: {monitored} forecast to reach {'the alarm band' if margin else 'the alarm level'} "
                     f"in **{fmt_hours(f['first_breach_hour'])}** ({when.strftime('%a %d %b %H:%M')}). "
                     f"Forecast peak {f['peak']:.1f}{unit} vs alarm level {limit:g}{unit}.")
        else:
            st.success(f"🟢 **NORMAL**: no breach forecast in the next {fmt_hours(outlook_hours)} "
                       f"(forecast peak {f['peak']:.1f}{unit}, alarm level {limit:g}{unit}).")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current", f"{ol['current']:.1f}{unit}")
        c2.metric(f"Forecast peak ({fmt_hours(outlook_hours)})", f"{f['peak']:.1f}{unit}",
                  delta=f"{f['peak'] - limit:+.1f} vs alarm", delta_color="inverse")
        c3.metric("Time to breach", fmt_hours(f["first_breach_hour"]))
        c4.metric("Hours in alarm band", f"{f['hours_above']}")

        reveal = st.toggle("🔍 Reveal what actually happened", value=False,
                           help="Show the real measurements after the outlook was issued.")
        st.plotly_chart(plot_outlook(fc, monitored, limit, margin, outlook_hours, reveal), use_container_width=True)
        st.plotly_chart(plot_risk_strip(ol["days"], limit, reveal), use_container_width=True)

        if reveal:
            a = ol["actual"]
            if ol["status"] == "in_breach":
                st.info("Already in breach at issue time. These days are excluded from the warning scores.")
            elif a["breach"] and f["breach"]:
                st.success(f"✅ **Correct warning.** The real breach came {fmt_hours(a['first_breach_hour'])} "
                           f"after the outlook was issued. That's the operator's lead time.")
            elif a["breach"]:
                st.error(f"❌ **Missed.** A breach happened at +{fmt_hours(a['first_breach_hour'])} "
                         f"(actual peak {a['peak']:.1f}{unit}) but wasn't forecast.")
            elif f["breach"]:
                st.warning(f"⚠️ **False alarm.** Actual peak stayed at {a['peak']:.1f}{unit}.")
            else:
                st.success(f"✅ **Correct all-clear.** Actual peak {a['peak']:.1f}{unit}.")

        if ol["status"] != "normal":
            with st.expander("Suggested operator response", expanded=True):
                st.markdown(
                    "- Check that cooling (fans, oil pumps) is available and switch it to forced mode ahead of the peak\n"
                    "- Plan load transfer to neighbouring transformers or defer flexible load around the forecast peak\n"
                    "- Increase monitoring frequency until the forecast returns to normal"
                )
                st.caption("Rule-based suggestions for illustration. Site procedures take precedence.")

        bulletin = (
            f"GridForecast outlook, issued {ol['issued_at']}\n"
            f"Signal: {monitored} ({channel_desc.get(monitored, '')}), alarm level {limit:g}{unit}, "
            f"margin {margin:g}{unit}, outlook {outlook_hours}h\n"
            f"Status: {ol['status'].upper()}\n"
            f"Current: {ol['current']:.1f}{unit} | Forecast peak: {f['peak']:.1f}{unit} | "
            f"Time to breach: {fmt_hours(f['first_breach_hour'])}\n\n30-day outlook (daily peak):\n"
            + "\n".join(f"  {d['date']}  {d['forecast_peak']:6.1f}{unit}  {d['level'].upper()}" for d in ol["days"])
        )
        st.download_button("📥 Download outlook bulletin", bulletin,
                           f"outlook_{ol['issued_at'][:10]}.txt", "text/plain")

# Tab: Warning Performance (the evidence)
with tab_perf:
    st.header("Would operators have been warned in time?")
    if not evaluation:
        st.info("This replay needs precomputed forecasts (app/precomputed/forecasts.npz). See app/README.md.")
        api_post("/warning/evaluate", **warning_params)  # surfaces the error message
    else:
        labels = evaluation["labels"]
        scores = evaluation["scores"]
        n_days = len(evaluation["records"])
        st.markdown(
            f"Replays **one {fmt_hours(outlook_hours)} outlook per day** across the test period "
            f"({n_days} days; {evaluation['skipped_in_breach']} skipped because the signal was already above "
            f"the alarm level). A day counts as a **breach** if {monitored} actually reached {limit:g}{unit} within "
            f"the outlook. FM-LLM is compared with simple rules an operator could use without a model."
        )
        if evaluation["events"] == 0:
            st.warning("No breaches happened at this alarm level. Lower it to evaluate warnings.")

        m = scores["fm_llm"]
        best_base = max((k for k in scores if k != "fm_llm"), key=lambda k: scores[k]["csi"] or 0)
        b = scores[best_base]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Breaches caught", f"{m['hits']} / {m['hits'] + m['misses']}",
                  delta=f"{m['hits'] - b['hits']:+d} vs best baseline")
        c2.metric("False alarms", m["false_alarms"],
                  delta=f"{m['false_alarms'] - b['false_alarms']:+d} vs best baseline", delta_color="inverse")
        c3.metric("Skill (CSI)", fmt_pct(m["csi"]),
                  help="Critical success index = hits / (hits + misses + false alarms)")
        c4.metric("Mean lead time", fmt_hours(m["mean_lead_hours"]),
                  help="Hours between the warning and the real breach, averaged over caught breaches")

        rows = []
        for k in evaluation["methods"]:
            s = scores[k]
            rows.append({
                "Method": labels[k],
                "Caught": s["hits"], "Missed": s["misses"], "False alarms": s["false_alarms"],
                "Hit rate": fmt_pct(s["hit_rate"]), "False alarm ratio": fmt_pct(s["false_alarm_ratio"]),
                "CSI (skill)": fmt_pct(s["csi"]), "Mean lead time": fmt_hours(s["mean_lead_hours"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.plotly_chart(plot_warning_timeline(evaluation), use_container_width=True)
        st.caption("Try other alarm levels, margins and outlooks in the sidebar: the replay reruns instantly.")

        st.subheader("Forecast accuracy vs. the same baselines")
        acc = api_post("/accuracy/compare", checkpoint=selected_checkpoint, channel=monitored)
        if acc:
            rows = []
            for k, by_h in acc["results"].items():
                row = {"Method": acc["labels"][k]}
                for h in ("96", "192", "336", "720"):
                    row[f"{monitored} MAE {h}h ({unit or 'units'})"] = by_h[h]["channel_mae"]
                row["All-channel MSE 720h"] = by_h["720"]["mse"]
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df.style.highlight_min(subset=df.columns[1:], color="lightgreen").format(precision=3),
                         use_container_width=True, hide_index=True)
            st.caption(f"Over the same {acc['num_windows']} daily outlooks. MSE is on the normalized scale "
                       "used by the paper.")

# Tab: Load What-if
with tab_whatif:
    st.header("Load what-if")
    st.markdown(
        f"How does the {monitored} outlook respond if recent load had been different? Scale the load "
        "channels over the **last 4 days** (96h) and compare against the base outlook."
    )
    st.caption("FM-LLM learned statistical patterns, not transformer physics, so treat this as a sensitivity "
               "check of the model rather than a physical simulation.")
    if not ckpt["live"]:
        st.info("What-ifs run the model live, which needs the checkpoint files and Llama-3.2-1B access "
                "(HF_TOKEN). This checkpoint is serving precomputed forecasts only.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            perturbations = {}
            for ch in [c for c in LOAD_CHANNELS if c in channels]:
                pct = st.slider(f"{ch} — {channel_desc.get(ch, ch)}", -50, 50, 0, step=5,
                                format="%d%%", key=f"pert_{ch}")
                if pct:
                    perturbations[ch] = pct / 100
            run = st.button("🔮 Run what-if", type="primary", use_container_width=True)
        with col2:
            if run and not perturbations:
                st.warning("Move at least one slider.")
            elif run:
                with st.spinner("Running base and what-if forecasts live..."):
                    scen = api_post("/scenario", window_idx=window_idx, horizon=max(outlook_hours, 96),
                                    checkpoint=selected_checkpoint, perturbations=perturbations)
                if scen:
                    i = scen["channels"].index(monitored)
                    base, new = column(scen["base"], i), column(scen["predictions"], i)
                    c1, c2 = st.columns(2)
                    c1.metric(f"Peak {monitored}", f"{new.max():.1f}{unit}", delta=f"{new.max() - base.max():+.2f}",
                              delta_color="inverse")
                    c2.metric("Time to breach", fmt_hours(first_breach(new, limit)),
                              delta=f"base: {fmt_hours(first_breach(base, limit))}", delta_color="off")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=scen["dates"], y=base, name="Base outlook", line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=scen["dates"], y=new, name="What-if", line=dict(width=2, dash="dot")))
                    fig.add_hline(y=limit, line_dash="dash", line_color=LEVEL_COLORS["red"])
                    fig.update_layout(height=420, hovermode="x unified", yaxis_title=f"{monitored} ({unit})")
                    st.plotly_chart(fig, use_container_width=True)

# Tab: Forecast Explorer
with tab_explore:
    st.header("Forecast explorer")
    col1, col2 = st.columns([2, 3])
    horizon = col1.radio("Horizon", HORIZONS, index=1, horizontal=True, format_func=lambda h: f"{h}h ({h // 24}d)")
    selected_channels = col2.multiselect(
        "Channels", channels, default=[c for c in ("OT", "HUFL") if c in channels],
        format_func=lambda c: f"{c} — {channel_desc.get(c, c)}",
    )
    result = fetch_forecast(window_idx, horizon, selected_checkpoint)
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE (normalized)", f"{result['metrics']['mse']:.3f}", help="Same scale as the paper's Table A.12")
        c2.metric("MAE (normalized)", f"{result['metrics']['mae']:.3f}")
        c3.metric("Source", result["source"])
        st.plotly_chart(plot_forecast(result, selected_channels), use_container_width=True)
        with st.expander("📋 Per-channel error & raw data"):
            st.dataframe(pd.DataFrame(result["per_channel_metrics"]).T.rename(columns=str.upper)
                         .style.format("{:.4f}"), use_container_width=True)
            df = pd.concat([
                pd.DataFrame(result["predictions"], columns=result["channels"]).add_suffix(" (forecast)"),
                pd.DataFrame(result["ground_truth"], columns=result["channels"]).add_suffix(" (actual)"),
            ], axis=1)
            df.insert(0, "date", result["dates"])
            st.dataframe(df, use_container_width=True)
            st.download_button("📥 Download CSV", df.to_csv(index=False),
                               f"forecast_{selected_checkpoint}_w{window_idx}_h{horizon}.csv", "text/csv")

# Tab: Model & Accuracy
with tab_model:
    st.header("Model & accuracy")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("How it works")
        st.markdown("""
        **FM-LLM: Frequency-Enhanced Mixture-of-Experts for Time Series Forecasting**

        1. **Fourier Embedding Module** (Eq. 7-11): 96-hour patches → MLP(512) → Fourier Analysis
           Network (periodic + trend) → Linear(2048)
        2. **Frozen Llama-3.2-1B**: reads 7 patches (28 days) per channel; weights frozen
        3. **FAN-MoE decoder** (Eq. 13-18): 2 shared FAN experts + 2 routed experts, top-1 gate
        4. **Autoregressive rollout**: sliding 7-patch window, 8 steps → 768h, truncated to the horizon
        """)
        st.subheader("Checkpoints")
        for c in checkpoints:
            st.markdown(f"- **{c['name']}**: {c['description']}")

    with col2:
        st.subheader("Full test set vs paper (Table A.12)")
        reported = fetch_results()
        paper = reported["paper"]
        rows = [
            {"Horizon": f"{h}h", "Checkpoint": name, "MSE": v[0], "MAE": v[1]}
            for name, by_h in reported["checkpoints"].items() for h, v in by_h.items()
        ] + [{"Horizon": f"{h}h", "Checkpoint": "paper", "MSE": v[0], "MAE": v[1]} for h, v in paper.items()]
        df_comp = pd.DataFrame(rows)
        for metric in ("MSE", "MAE"):
            st.markdown(f"**{metric}** (normalized, lower is better)")
            pivot = df_comp.pivot(index="Horizon", columns="Checkpoint", values=metric)
            ours = [c for c in pivot.columns if c != "paper"]
            st.dataframe(pivot.style.highlight_min(axis=1, subset=ours, color="lightgreen").format("{:.4f}"),
                         use_container_width=True)
        st.info("**Key finding:** per-window instance normalization cut 720h MSE from 0.6381 to 0.4721 "
                "(−26%), shrinking the gap to the paper from ~61% to ~19%.")

    st.subheader("Backtest")
    bcol1, bcol2 = st.columns([1, 3])
    with bcol1:
        max_bt = ckpt["precomputed_windows"] or 200
        bt_windows = st.slider("Number of windows", 1, min(200, max_bt), min(50, max_bt))
        if not ckpt["precomputed_windows"]:
            st.caption("⚠️ No precomputed forecasts: every window runs live (~10s+ each on CPU).")
        run_bt = st.button("📊 Run backtest", type="primary", use_container_width=True)
    with bcol2:
        if run_bt:
            with st.spinner(f"Backtesting {bt_windows} windows..."):
                bt = api_post("/backtest", checkpoint=selected_checkpoint, num_windows=bt_windows,
                              horizons=list(HORIZONS))
            if bt:
                st.plotly_chart(plot_backtest_results(bt["results"]), use_container_width=True)
                st.caption(f"{len(bt['windows'])} windows. The full-test-set table above uses all 2161.")

st.divider()
st.caption("GridForecast v0.3.0 | FM-LLM reproduction | Streamlit + FastAPI")
