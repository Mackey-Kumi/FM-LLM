"""
Streamlit UI for GridForecast - Electricity Demand Forecasting Application
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, List, Optional
import json

# Page config
st.set_page_config(
    page_title="GridForecast - Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API base URL
API_BASE = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def fetch_health():
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_checkpoints():
    try:
        response = requests.get(f"{API_BASE}/checkpoints", timeout=5)
        return response.json()["checkpoints"]
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_channels():
    try:
        response = requests.get(f"{API_BASE}/channels", timeout=5)
        return response.json()["channels"]
    except Exception:
        return []


def fetch_forecast(window_idx: int, horizon: int, checkpoint: str):
    try:
        response = requests.post(
            f"{API_BASE}/forecast",
            json={"window_idx": window_idx, "horizon": horizon, "checkpoint": checkpoint},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def fetch_all_horizons(window_idx: int, checkpoint: str):
    try:
        response = requests.post(
            f"{API_BASE}/forecast/all_horizons",
            params={"window_idx": window_idx, "checkpoint": checkpoint},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def fetch_scenario(window_idx: int, horizon: int, perturbations: Dict[str, float]):
    try:
        response = requests.post(
            f"{API_BASE}/scenario",
            json={"window_idx": window_idx, "horizon": horizon, "perturbations": perturbations},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def fetch_backtest(num_windows: int, horizons: List[int]):
    try:
        response = requests.post(
            f"{API_BASE}/backtest",
            json={"num_windows": num_windows, "horizons": horizons},
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def fetch_demo_window(window_idx: int):
    try:
        response = requests.get(f"{API_BASE}/demo/window/{window_idx}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def plot_forecast(predictions, ground_truth, channels, horizon, selected_channels):
    """Create forecast vs ground truth plot"""
    fig = go.Figure()
    
    time_steps = list(range(horizon))
    
    for i, ch in enumerate(channels):
        if ch not in selected_channels:
            continue
            
        pred = [p[i] for p in predictions]
        truth = [g[i] for g in ground_truth]
        
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=truth,
            mode='lines',
            name=f'{ch} (Actual)',
            line=dict(width=2, dash='solid'),
            opacity=0.7,
        ))
        
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=pred,
            mode='lines',
            name=f'{ch} (Forecast)',
            line=dict(width=2, dash='dash'),
            opacity=0.9,
        ))
    
    fig.update_layout(
        title=f"Forecast vs Actual (Horizon: {horizon}h)",
        xaxis_title="Hours Ahead",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_metrics_comparison(results):
    """Plot MSE/MAE across horizons"""
    horizons = sorted([int(h) for h in results.keys()])
    mse_vals = [results[str(h)]["mse"] for h in horizons]
    mae_vals = [results[str(h)]["mae"] for h in horizons]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=horizons, y=mse_vals, name="MSE", marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=horizons, y=mae_vals, name="MAE", marker_color='#ff7f0e'))
    
    fig.update_layout(
        title="Error Metrics Across Horizons",
        xaxis_title="Horizon (hours)",
        yaxis_title="Error",
        barmode='group',
        height=400,
    )
    return fig


def plot_backtest_results(backtest_results):
    """Plot backtest aggregate metrics"""
    horizons = sorted([int(h) for h in backtest_results.keys()])
    
    fig = go.Figure()
    
    for metric in ['mse', 'mae']:
        means = [backtest_results[str(h)][f"{metric}_mean"] for h in horizons]
        stds = [backtest_results[str(h)][f"{metric}_std"] for h in horizons]
        
        fig.add_trace(go.Scatter(
            x=horizons,
            y=means,
            mode='lines+markers',
            name=metric.upper(),
            error_y=dict(type='data', array=stds, visible=True),
        ))
    
    fig.update_layout(
        title="Backtest Results (Mean ± Std across windows)",
        xaxis_title="Horizon (hours)",
        yaxis_title="Error",
        height=400,
    )
    return fig


def plot_scenario_comparison(base_pred, scenario_pred, channels, horizon, selected_channels):
    """Plot base forecast vs scenario forecast"""
    fig = go.Figure()
    time_steps = list(range(horizon))
    
    for i, ch in enumerate(channels):
        if ch not in selected_channels:
            continue
            
        base = [p[i] for p in base_pred]
        scen = [p[i] for p in scenario_pred]
        
        fig.add_trace(go.Scatter(
            x=time_steps, y=base,
            mode='lines', name=f'{ch} (Base)',
            line=dict(width=2, dash='solid'),
        ))
        fig.add_trace(go.Scatter(
            x=time_steps, y=scen,
            mode='lines', name=f'{ch} (Scenario)',
            line=dict(width=2, dash='dot'),
        ))
    
    fig.update_layout(
        title="Scenario Analysis: Base vs Perturbed Forecast",
        xaxis_title="Hours Ahead",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
    )
    return fig


# Sidebar
with st.sidebar:
    st.markdown('<div class="main-header">⚡ GridForecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">FM-LLM Electricity Demand Forecasting</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Health check
    health = fetch_health()
    if health:
        st.success(f"✅ API Connected")
        st.caption(f"Model: {health['model']} | Device: {health['device']} | Windows: {health['available_windows']}")
    else:
        st.error("❌ API Disconnected")
        st.caption("Start the API server: `python -m app.api.main`")
    
    st.divider()
    
    # Global controls
    checkpoints = fetch_checkpoints()
    checkpoint_names = [c["name"] for c in checkpoints]
    selected_checkpoint = st.selectbox("Model Checkpoint", checkpoint_names, index=0)
    
    channels = fetch_channels()
    selected_channels = st.multiselect("Channels to Display", channels, default=channels[:3])
    
    st.divider()
    st.markdown("### Quick Links")
    st.markdown("- [API Docs](http://localhost:8000/docs)")
    st.markdown("- [GitHub Repo](https://github.com)")

# Main content
st.markdown('<div class="main-header">⚡ GridForecast</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">From 7 days of history → 30 days of hourly electricity demand forecasts</div>', unsafe_allow_html=True)

if not health:
    st.error("⚠️ Backend API not running. Please start it with: `python -m app.api.main`")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Forecast", 
    "🔮 Scenario Simulator", 
    "📈 Backtest & Metrics", 
    "🚨 Alerts & Monitoring",
    "🔬 Model Explorer"
])

# Tab 1: Live Forecast
with tab1:
    st.header("Live Forecast Dashboard")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        max_windows = health['available_windows']
        window_idx = st.number_input(
            "Test Window Index", 
            min_value=0, 
            max_value=max_windows - 1, 
            value=0,
            help=f"Select a test window (0 to {max_windows-1})"
        )
    
    with col2:
        horizon = st.selectbox("Forecast Horizon", HORIZONS, index=1)
    
    with col3:
        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Generating {horizon}h forecast..."):
                result = fetch_forecast(window_idx, horizon, selected_checkpoint)
                if result:
                    st.session_state['last_forecast'] = result
                    st.success("Forecast generated!")
    
    # Display forecast
    if 'last_forecast' in st.session_state:
        result = st.session_state['last_forecast']
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{result["metrics"]["mse"]:.4f}</div><div class="metric-label">MSE</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{result["metrics"]["mae"]:.4f}</div><div class="metric-label">MAE</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{horizon}</div><div class="metric-label">Horizon (h)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{window_idx}</div><div class="metric-label">Window Index</div></div>', unsafe_allow_html=True)
        
        # Plot
        fig = plot_forecast(
            result["predictions"], 
            result["ground_truth"], 
            result["channels"], 
            result["horizon"],
            selected_channels
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            df_pred = pd.DataFrame(result["predictions"], columns=result["channels"])
            df_truth = pd.DataFrame(result["ground_truth"], columns=result["channels"])
            df_combined = pd.concat([
                df_pred.add_suffix(' (Pred)'),
                df_truth.add_suffix(' (Actual)')
            ], axis=1)
            st.dataframe(df_combined, use_container_width=True)
            
            # Download button
            csv = df_combined.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"forecast_window{window_idx}_h{horizon}.csv",
                "text/csv"
            )

# Tab 2: Scenario Simulator
with tab2:
    st.header("Scenario Simulator")
    st.markdown("Modify input channels (e.g., temperature, load) and see how forecasts change.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Setup")
        scen_window = st.number_input("Window Index", min_value=0, max_value=max_windows-1, value=0, key="scen_window")
        scen_horizon = st.selectbox("Horizon", HORIZONS, index=1, key="scen_horizon")
        
        st.markdown("**Channel Perturbations**")
        st.caption("Enter percentage change (e.g., 0.05 = +5%, -0.1 = -10%)")
        
        perturbations = {}
        for ch in channels:
            val = st.number_input(
                f"{ch}", 
                min_value=-1.0, 
                max_value=1.0, 
                value=0.0, 
                step=0.01,
                key=f"pert_{ch}"
            )
            if val != 0:
                perturbations[ch] = val
        
        if st.button("🔮 Run Scenario", type="primary", use_container_width=True):
            if not perturbations:
                st.warning("No perturbations set. Adjust at least one channel.")
            else:
                with st.spinner("Running scenario..."):
                    # Get base forecast first
                    base = fetch_forecast(scen_window, scen_horizon, selected_checkpoint)
                    scen = fetch_scenario(scen_window, scen_horizon, perturbations)
                    
                    if base and scen:
                        st.session_state['scenario_base'] = base
                        st.session_state['scenario_result'] = scen
                        st.success("Scenario complete!")
    
    with col2:
        if 'scenario_base' in st.session_state and 'scenario_result' in st.session_state:
            base = st.session_state['scenario_base']
            scen = st.session_state['scenario_result']
            
            fig = plot_scenario_comparison(
                base["predictions"],
                scen["predictions"],
                channels,
                scen_horizon,
                selected_channels
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Difference table
            st.subheader("Impact Summary")
            diff_data = []
            for i, ch in enumerate(channels):
                if ch in selected_channels:
                    base_vals = np.array([p[i] for p in base["predictions"]])
                    scen_vals = np.array([p[i] for p in scen["predictions"]])
                    diff = scen_vals - base_vals
                    diff_data.append({
                        "Channel": ch,
                        "Mean Change": f"{diff.mean():.4f}",
                        "Max Change": f"{diff.max():.4f}",
                        "Min Change": f"{diff.min():.4f}",
                    })
            st.dataframe(pd.DataFrame(diff_data), use_container_width=True)

# Tab 3: Backtest & Metrics
with tab3:
    st.header("Backtest & Model Metrics")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Backtest Configuration")
        bt_windows = st.slider("Number of Windows", 10, 200, 50, step=10)
        bt_horizons = st.multiselect("Horizons", HORIZONS, default=list(HORIZONS))
        
        if st.button("📊 Run Backtest", type="primary", use_container_width=True):
            with st.spinner(f"Running backtest on {bt_windows} windows... (this may take a minute)"):
                result = fetch_backtest(bt_windows, bt_horizons)
                if result:
                    st.session_state['backtest'] = result
                    st.success("Backtest complete!")
    
    with col2:
        if 'backtest' in st.session_state:
            bt = st.session_state['backtest']['results']
            
            # Summary table
            st.subheader("Aggregate Metrics")
            summary_data = []
            for h in sorted([int(k) for k in bt.keys()]):
                h_str = str(h)
                summary_data.append({
                    "Horizon": f"{h}h",
                    "MSE (mean)": f"{bt[h_str]['mse_mean']:.4f}",
                    "MSE (std)": f"{bt[h_str]['mse_std']:.4f}",
                    "MAE (mean)": f"{bt[h_str]['mae_mean']:.4f}",
                    "MAE (std)": f"{bt[h_str]['mae_std']:.4f}",
                    "Windows": bt[h_str]['num_windows'],
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Plot
            fig = plot_backtest_results(bt)
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare with paper
            st.subheader("Comparison with Paper (Table A.12)")
            paper_mse = {192: 0.377, 336: 0.395, 720: 0.397}
            paper_mae = {192: 0.403, 336: 0.415, 720: 0.429}
            
            comp_data = []
            for h in [192, 336, 720]:
                if str(h) in bt:
                    comp_data.append({
                        "Horizon": f"{h}h",
                        "Our MSE": f"{bt[str(h)]['mse_mean']:.4f}",
                        "Paper MSE": f"{paper_mse[h]:.4f}",
                        "Gap MSE": f"{bt[str(h)]['mse_mean'] - paper_mse[h]:.4f}",
                        "Our MAE": f"{bt[str(h)]['mae_mean']:.4f}",
                        "Paper MAE": f"{paper_mae[h]:.4f}",
                        "Gap MAE": f"{bt[str(h)]['mae_mean'] - paper_mae[h]:.4f}",
                    })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# Tab 4: Alerts & Monitoring
with tab4:
    st.header("Alerts & Monitoring")
    st.markdown("Configure thresholds and monitor for anomalous forecasts.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Alert Configuration")
        
        alert_channel = st.selectbox("Monitor Channel", channels, index=channels.index('OT'))
        
        alert_type = st.radio("Alert Type", ["Threshold", "Anomaly (z-score)"])
        
        if alert_type == "Threshold":
            upper = st.number_input("Upper Threshold", value=50.0, step=1.0)
            lower = st.number_input("Lower Threshold", value=10.0, step=1.0)
        else:
            z_threshold = st.number_input("Z-Score Threshold", value=3.0, step=0.5)
            window = st.number_input("Rolling Window (hours)", value=168, step=24)
        
        st.divider()
        
        # Check alerts on last forecast
        if 'last_forecast' in st.session_state:
            result = st.session_state['last_forecast']
            ch_idx = channels.index(alert_channel)
            pred_vals = [p[ch_idx] for p in result["predictions"]]
            
            st.subheader("Current Forecast Alerts")
            
            if alert_type == "Threshold":
                violations = [(i, v) for i, v in enumerate(pred_vals) if v > upper or v < lower]
                if violations:
                    st.error(f"⚠️ {len(violations)} threshold violations!")
                    for idx, val in violations[:10]:
                        st.caption(f"Hour {idx}: {val:.2f} ({'HIGH' if val > upper else 'LOW'})")
                else:
                    st.success("✅ No threshold violations")
            else:
                mean_val = np.mean(pred_vals)
                std_val = np.std(pred_vals)
                anomalies = [(i, v) for i, v in enumerate(pred_vals) if abs(v - mean_val) / (std_val + 1e-8) > z_threshold]
                if anomalies:
                    st.error(f"⚠️ {len(anomalies)} anomalies detected!")
                    for idx, val in anomalies[:10]:
                        z = abs(val - mean_val) / (std_val + 1e-8)
                        st.caption(f"Hour {idx}: {val:.2f} (z={z:.2f})")
                else:
                    st.success("✅ No anomalies detected")
    
    with col2:
        if 'last_forecast' in st.session_state:
            result = st.session_state['last_forecast']
            ch_idx = channels.index(alert_channel)
            pred_vals = [p[ch_idx] for p in result["predictions"]]
            truth_vals = [g[ch_idx] for g in result["ground_truth"]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=truth_vals, name="Actual", line=dict(color='gray')))
            fig.add_trace(go.Scatter(y=pred_vals, name="Forecast", line=dict(color='blue', dash='dash')))
            
            if alert_type == "Threshold":
                fig.add_hline(y=upper, line_dash="dash", line_color="red", annotation_text="Upper")
                fig.add_hline(y=lower, line_dash="dash", line_color="red", annotation_text="Lower")
            else:
                mean_val = np.mean(pred_vals)
                std_val = np.std(pred_vals)
                fig.add_hline(y=mean_val + z_threshold*std_val, line_dash="dash", line_color="orange")
                fig.add_hline(y=mean_val - z_threshold*std_val, line_dash="dash", line_color="orange")
            
            fig.update_layout(title=f"{alert_channel} Forecast with Alert Thresholds", height=400)
            st.plotly_chart(fig, use_container_width=True)

# Tab 5: Model Explorer
with tab5:
    st.header("Model Explorer")
    st.markdown("Understand the FM-LLM architecture and checkpoint performance.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Architecture")
        st.markdown("""
        **FM-LLM: Frequency-Enhanced Mixture-of-Experts for Time Series Forecasting**
        
        **Pipeline:**
        1. **Fourier Embedding Module** (Eq. 7-11)
           - Patch length: 96 → MLP(512) → FAN → Linear(2048)
           - Frequency Analysis Network (FAN) decomposes periodic/trend
        
        2. **Frozen Llama-3.2-1B Backbone**
           - 2048-dim embeddings, frozen weights
           - Processes 7 tokens (672 steps) as sequence
        
        3. **FAN-MoE Decoder** (Eq. 13-18)
           - 2 Shared FAN Experts
           - 2 Routed Experts + Top-1 Gate
           - Load-balanced routing with bias correction
        
        4. **Autoregressive Rollout**
           - Sliding 7-token window (672 steps)
           - 8 steps → 768h → truncate to target horizon
        """)
        
        st.subheader("Checkpoints")
        for ckpt in checkpoints:
            with st.expander(f"{ckpt['name']}"):
                st.write(ckpt['description'])
    
    with col2:
        st.subheader("Checkpoint Comparison (ETTh1, input=672)")
        
        # Recreate the table from README
        comparison_data = {
            "Horizon": ["192", "192", "192", "192", "336", "336", "336", "336", "720", "720", "720", "720"],
            "Checkpoint": ["No Dropout", "Dropout", "Dropout (Longer)", "Instance Norm"] * 3,
            "MSE": [0.5089, 0.4417, 0.4292, 0.3945, 0.5423, 0.4858, 0.4641, 0.4149, 0.6381, 0.5894, 0.5422, 0.4721],
            "MAE": [0.4882, 0.4571, 0.4445, 0.4225, 0.5158, 0.4925, 0.4730, 0.4402, 0.5792, 0.5661, 0.5350, 0.4839],
        }
        df_comp = pd.DataFrame(comparison_data)
        
        # Pivot for display
        pivot_mse = df_comp.pivot(index="Horizon", columns="Checkpoint", values="MSE")
        pivot_mae = df_comp.pivot(index="Horizon", columns="Checkpoint", values="MAE")
        
        st.markdown("**MSE**")
        st.dataframe(pivot_mse.style.highlight_min(axis=1, color='lightgreen'), use_container_width=True)
        
        st.markdown("**MAE**")
        st.dataframe(pivot_mae.style.highlight_min(axis=1, color='lightgreen'), use_container_width=True)
        
        # Key insight
        st.info("""
        **Key Finding:** Instance Normalization (per-window) closes 70% of the gap to paper at 720h.
        - No dropout → Instance Norm: MSE 0.6381 → 0.4721 at 720h (**26% improvement**)
        - Residual gap at 720h: ~19% vs paper (down from ~37%)
        """)

# Footer
st.divider()
st.caption("GridForecast v0.1.0 | FM-LLM Reproduction | Built with Streamlit + FastAPI")