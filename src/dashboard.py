"""
NIDS Dashboard - Real-Time Network Intrusion Detection Monitoring.

Professional dashboard con:
- Real-time statistics
- Attack classification breakdown
- Blocked IPs management
- Performance metrics
- Historical trends

Usage:
    streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import json

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SNIFFER_OUTPUT = DATA_DIR / "sniffer_output"
LOGS_DIR = BASE_DIR / "logs"

# Colors
COLORS = {
    'benign': '#2ecc71',
    'attack': '#e74c3c',
    'blocked': '#c0392b',
    'primary': '#3498db',
    'secondary': '#95a5a6',
    'warning': '#f39c12'
}


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=60)
def load_latest_predictions():
    """Carica le predizioni più recenti."""
    if not SNIFFER_OUTPUT.exists():
        return None
    
    # Trova file più recente
    csv_files = list(SNIFFER_OUTPUT.glob('predictions_*.csv'))
    
    if not csv_files:
        return None
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    df = pd.read_csv(latest_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df, latest_file


@st.cache_data(ttl=300)
def load_model_comparison():
    """Carica risultati comparison se disponibili."""
    comparison_file = BASE_DIR / "docs" / "Comparison" / "comparison_results.csv"
    
    if comparison_file.exists():
        return pd.read_csv(comparison_file)
    
    return None


def load_available_models():
    """Lista modelli disponibili."""
    models = []
    
    for algo_dir in MODELS_DIR.iterdir():
        if algo_dir.is_dir():
            for model_file in algo_dir.glob('*.pkl'):
                models.append({
                    'algorithm': algo_dir.name,
                    'file': model_file.name,
                    'path': str(model_file),
                    'size_mb': model_file.stat().st_size / 1024**2
                })
    
    return pd.DataFrame(models) if models else None


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(df):
    """Calcola metriche aggregate."""
    if df is None or len(df) == 0:
        return None
    
    total_flows = len(df)
    attacks = df[df['is_attack'] == True]
    benign = df[df['is_attack'] == False]
    blocked = df[df['blocked'] == True]
    
    metrics = {
        'total_flows': total_flows,
        'benign_count': len(benign),
        'attack_count': len(attacks),
        'blocked_count': len(blocked),
        'attack_rate': len(attacks) / total_flows * 100 if total_flows > 0 else 0,
        'block_rate': len(blocked) / len(attacks) * 100 if len(attacks) > 0 else 0,
        'avg_confidence': df['confidence'].mean(),
        'unique_ips': df['src_ip'].nunique()
    }
    
    return metrics


def get_attack_breakdown(df):
    """Breakdown attacchi per tipo."""
    if df is None or len(df) == 0:
        return None
    
    attacks = df[df['is_attack'] == True]
    
    if len(attacks) == 0:
        return None
    
    breakdown = attacks['predicted_class'].value_counts()
    
    return breakdown


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_attack_pie(df):
    """Pie chart: Benign vs Attacks."""
    if df is None or len(df) == 0:
        return None
    
    counts = df['is_attack'].value_counts()
    labels = ['Benign', 'Attack']
    values = [counts.get(False, 0), counts.get(True, 0)]
    colors = [COLORS['benign'], COLORS['attack']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textposition='inside',
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title="Traffic Classification",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_attack_types_bar(breakdown):
    """Bar chart: Attack types."""
    if breakdown is None:
        return None
    
    fig = go.Figure(data=[go.Bar(
        x=breakdown.index,
        y=breakdown.values,
        marker=dict(color=COLORS['attack']),
        text=breakdown.values,
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Attack Types Distribution",
        xaxis_title="Attack Class",
        yaxis_title="Count",
        height=400,
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig


def plot_timeline(df):
    """Timeline: Attacks over time."""
    if df is None or len(df) == 0:
        return None
    
    df_sorted = df.sort_values('timestamp')
    
    # Resample per ora
    df_sorted['hour'] = df_sorted['timestamp'].dt.floor('H')
    
    timeline = df_sorted.groupby(['hour', 'is_attack']).size().reset_index(name='count')
    
    fig = px.line(
        timeline,
        x='hour',
        y='count',
        color='is_attack',
        color_discrete_map={True: COLORS['attack'], False: COLORS['benign']},
        labels={'hour': 'Time', 'count': 'Flow Count', 'is_attack': 'Type'}
    )
    
    fig.update_layout(
        title="Traffic Timeline",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_confidence_distribution(df):
    """Histogram: Confidence distribution."""
    if df is None or len(df) == 0:
        return None
    
    fig = go.Figure()
    
    # Benign
    benign = df[df['is_attack'] == False]['confidence']
    fig.add_trace(go.Histogram(
        x=benign,
        name='Benign',
        marker_color=COLORS['benign'],
        opacity=0.7
    ))
    
    # Attacks
    attacks = df[df['is_attack'] == True]['confidence']
    fig.add_trace(go.Histogram(
        x=attacks,
        name='Attack',
        marker_color=COLORS['attack'],
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=400,
        barmode='overlay'
    )
    
    return fig


def plot_top_ips(df, n=10):
    """Top N IP addresses."""
    if df is None or len(df) == 0:
        return None
    
    attacks = df[df['is_attack'] == True]
    
    if len(attacks) == 0:
        return None
    
    top_ips = attacks['src_ip'].value_counts().head(n)
    
    fig = go.Figure(data=[go.Bar(
        x=top_ips.values,
        y=top_ips.index,
        orientation='h',
        marker=dict(color=COLORS['attack']),
        text=top_ips.values,
        textposition='outside'
    )])
    
    fig.update_layout(
        title=f"Top {n} Malicious IP Addresses",
        xaxis_title="Attack Count",
        yaxis_title="IP Address",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_model_comparison(df_comp):
    """Confronto performance modelli."""
    if df_comp is None:
        return None
    
    fig = go.Figure()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=df_comp['algorithm'] + ' (' + df_comp['dataset'] + ')',
            y=df_comp[metric],
            text=df_comp[metric].round(4),
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model (Dataset)",
        yaxis_title="Score",
        height=500,
        barmode='group',
        yaxis=dict(range=[0.85, 1.0])
    )
    
    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    # Header
    st.title("🛡️ NIDS Real-Time Dashboard")
    st.markdown("**Network Intrusion Detection System** - CICIoT2023")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Interval (seconds)", 5, 60, 10)
            st.info(f"Auto-refreshing every {refresh_interval}s")
        
        st.markdown("---")
        
        # Info
        st.subheader("📊 About")
        st.info(
            "This dashboard displays real-time network intrusion detection "
            "statistics from the NIDS sniffer.\n\n"
            "Data is loaded from the sniffer output directory."
        )
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_latest_predictions()
    
    if data is None:
        st.warning("⚠️ No prediction data available yet.")
        st.info(
            "Run the sniffer first:\n"
            "```bash\n"
            "sudo python src/sniffer.py --interface eth0 --model models/best_model.pkl\n"
            "```"
        )
        return
    
    df, data_file = data
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    breakdown = get_attack_breakdown(df)
    
    # Display file info
    st.success(f"📂 Data loaded: `{data_file.name}` ({len(df):,} flows)")
    
    st.markdown("---")
    
    # =============================================================================
    # METRICS ROW
    # =============================================================================
    
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Flows",
            f"{metrics['total_flows']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Benign",
            f"{metrics['benign_count']:,}",
            delta=f"{metrics['attack_rate']:.1f}% attacks",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Attacks Detected",
            f"{metrics['attack_count']:,}",
            delta=f"{metrics['attack_rate']:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "IPs Blocked",
            f"{metrics['blocked_count']:,}",
            delta=f"{metrics['block_rate']:.1f}% of attacks",
            delta_color="normal"
        )
    
    with col5:
        st.metric(
            "Avg Confidence",
            f"{metrics['avg_confidence']:.3f}",
            delta=None
        )
    
    st.markdown("---")
    
    # =============================================================================
    # CHARTS ROW 1
    # =============================================================================
    
    st.subheader("📊 Traffic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = plot_attack_pie(df)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if breakdown is not None and len(breakdown) > 0:
            fig_bar = plot_attack_types_bar(breakdown)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No attacks detected yet")
    
    # =============================================================================
    # TIMELINE
    # =============================================================================
    
    st.subheader("⏱️ Timeline")
    
    fig_timeline = plot_timeline(df)
    if fig_timeline:
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # =============================================================================
    # CHARTS ROW 2
    # =============================================================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Confidence Distribution")
        fig_conf = plot_confidence_distribution(df)
        if fig_conf:
            st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        st.subheader("🚫 Top Malicious IPs")
        fig_ips = plot_top_ips(df, n=10)
        if fig_ips:
            st.plotly_chart(fig_ips, use_container_width=True)
        else:
            st.info("No attacks detected yet")
    
    st.markdown("---")
    
    # =============================================================================
    # DETAILED TABLES
    # =============================================================================
    
    st.subheader("📋 Detailed Data")
    
    tab1, tab2, tab3 = st.tabs(["Recent Detections", "Blocked IPs", "All Flows"])
    
    with tab1:
        st.markdown("**Most Recent Attack Detections**")
        attacks = df[df['is_attack'] == True].sort_values('timestamp', ascending=False)
        
        if len(attacks) > 0:
            display_cols = ['timestamp', 'flow_id', 'src_ip', 'dst_ip', 
                          'predicted_class', 'confidence', 'blocked']
            st.dataframe(
                attacks[display_cols].head(50),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No attacks detected")
    
    with tab2:
        st.markdown("**Blocked IP Addresses**")
        blocked = df[df['blocked'] == True].sort_values('timestamp', ascending=False)
        
        if len(blocked) > 0:
            # Aggregate by IP
            blocked_summary = blocked.groupby('src_ip').agg({
                'timestamp': ['min', 'max', 'count'],
                'predicted_class': lambda x: ', '.join(x.unique()),
                'confidence': 'mean'
            }).reset_index()
            
            blocked_summary.columns = ['IP Address', 'First Seen', 'Last Seen', 
                                      'Attack Count', 'Attack Types', 'Avg Confidence']
            
            st.dataframe(
                blocked_summary,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No IPs blocked")
    
    with tab3:
        st.markdown("**All Traffic Flows**")
        display_cols = ['timestamp', 'flow_id', 'src_ip', 'dst_ip', 'protocol',
                       'predicted_class', 'confidence', 'is_attack', 'blocked']
        
        st.dataframe(
            df[display_cols].sort_values('timestamp', ascending=False).head(100),
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # =============================================================================
    # MODEL INFO
    # =============================================================================
    
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Models**")
        df_models = load_available_models()
        
        if df_models is not None:
            st.dataframe(
                df_models[['algorithm', 'file', 'size_mb']],
                use_container_width=True,
                height=200
            )
        else:
            st.info("No models found")
    
    with col2:
        st.markdown("**Model Comparison**")
        df_comp = load_model_comparison()
        
        if df_comp is not None:
            # Show best
            best = df_comp.loc[df_comp['accuracy'].idxmax()]
            st.success(
                f"**Best Model**: {best['algorithm']} ({best['dataset']})\n\n"
                f"- Accuracy: {best['accuracy']:.4f}\n"
                f"- Precision: {best['precision']:.4f}\n"
                f"- Recall: {best['recall']:.4f}"
            )
        else:
            st.info("Run comparison first: `python src/compare_models.py`")
    
    # Model comparison chart
    if df_comp is not None:
        st.markdown("**Performance Comparison**")
        fig_comp = plot_model_comparison(df_comp)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == '__main__':
    main()