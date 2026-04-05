import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Sector Intelligence Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4f8bf9;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #4f8bf9;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Generation (realistic synthetic India macro data) ─────────────────────
@st.cache_data
def load_macro_data():
    np.random.seed(42)
    dates = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    n = len(dates)

    # GDP Growth with COVID shock
    gdp = 7 + np.random.normal(0, 0.5, n)
    gdp[60:63] -= 8  # COVID crash Q1-Q3 2020
    gdp[63:66] += 5  # recovery

    # Inflation (CPI)
    cpi = 5 + np.random.normal(0, 0.8, n)
    cpi[70:80] += 2.5  # post-COVID inflation spike

    # Repo Rate
    repo = np.where(pd.Series(range(n)) < 65, 5.5, 4.0)
    repo = np.where(pd.Series(range(n)) > 75, 6.5, repo)

    # IIP (Industrial Production)
    iip = 3 + np.random.normal(0, 1.2, n)
    iip[60:63] -= 15
    iip[63:70] += 6

    # INR/USD
    inr = 65 + np.cumsum(np.random.normal(0.1, 0.3, n))

    # Trade deficit (USD bn)
    trade = -15 + np.random.normal(0, 3, n)

    df = pd.DataFrame({
        'Date': dates,
        'GDP_Growth': np.round(gdp, 2),
        'CPI_Inflation': np.round(cpi, 2),
        'Repo_Rate': np.round(repo, 2),
        'IIP_Growth': np.round(iip, 2),
        'INR_USD': np.round(inr, 2),
        'Trade_Deficit_USD_bn': np.round(trade, 2)
    })
    return df

@st.cache_data
def load_sector_data():
    np.random.seed(99)
    dates = pd.date_range(start='2015-01-01', end='2024-12-01', freq='MS')
    n = len(dates)
    sectors = {
        'IT': 12 + np.random.normal(0, 2, n),
        'Banking': 8 + np.random.normal(0, 3, n),
        'Pharma': 10 + np.random.normal(0, 2.5, n),
        'FMCG': 7 + np.random.normal(0, 1.5, n),
        'Auto': 6 + np.random.normal(0, 4, n),
        'Infra': 9 + np.random.normal(0, 3, n),
    }
    df = pd.DataFrame(sectors, index=dates).reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    return df

@st.cache_data
def run_stationarity_test(series):
    result = adfuller(series.dropna())
    return {
        'ADF Statistic': round(result[0], 4),
        'p-value': round(result[1], 4),
        'Stationary': 'Yes ✅' if result[1] < 0.05 else 'No ❌'
    }

@st.cache_data
def run_decomposition(series, dates):
    result = seasonal_decompose(series, model='additive', period=12)
    df = pd.DataFrame({
        'Date': dates,
        'Observed': result.observed,
        'Trend': result.trend,
        'Seasonal': result.seasonal,
        'Residual': result.resid
    })
    return df

# ── Load data ──────────────────────────────────────────────────────────────────
macro_df = load_macro_data()
sector_df = load_sector_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg", width=80)
st.sidebar.title("🔍 Controls")

date_range = st.sidebar.date_input(
    "Date Range",
    value=[macro_df['Date'].min(), macro_df['Date'].max()]
)

selected_indicator = st.sidebar.selectbox(
    "Primary Macro Indicator",
    ['GDP_Growth', 'CPI_Inflation', 'Repo_Rate', 'IIP_Growth', 'INR_USD', 'Trade_Deficit_USD_bn']
)

selected_sectors = st.sidebar.multiselect(
    "Sectors to Compare",
    ['IT', 'Banking', 'Pharma', 'FMCG', 'Auto', 'Infra'],
    default=['IT', 'Banking', 'Auto']
)

show_analysis = st.sidebar.checkbox("Show Statistical Analysis", value=True)

# ── Filter data ────────────────────────────────────────────────────────────────
filtered = macro_df[
    (macro_df['Date'] >= pd.Timestamp(date_range[0])) &
    (macro_df['Date'] <= pd.Timestamp(date_range[1]))
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🇮🇳 India Sector Intelligence Dashboard")
st.caption("Macroeconomic & Sector-level analysis | Data: RBI, MOSPI, NSE (Simulated)")

st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg GDP Growth", f"{filtered['GDP_Growth'].mean():.2f}%", f"{filtered['GDP_Growth'].iloc[-1] - filtered['GDP_Growth'].iloc[-2]:.2f}%")
k2.metric("Avg CPI", f"{filtered['CPI_Inflation'].mean():.2f}%")
k3.metric("Current Repo Rate", f"{filtered['Repo_Rate'].iloc[-1]:.2f}%")
k4.metric("Avg IIP Growth", f"{filtered['IIP_Growth'].mean():.2f}%")
k5.metric("INR/USD", f"₹{filtered['INR_USD'].iloc[-1]:.2f}")

st.markdown("---")

# ── Row 1: Macro trend + Sector comparison ─────────────────────────────────────
c1, c2 = st.columns([1.2, 1])

with c1:
    st.markdown(f'<div class="section-header">📈 {selected_indicator.replace("_", " ")} Over Time</div>', unsafe_allow_html=True)
    fig1 = px.area(
        filtered, x='Date', y=selected_indicator,
        color_discrete_sequence=['#4f8bf9'],
        template='plotly_dark'
    )
    fig1.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown('<div class="section-header">🏭 Sector Revenue Growth (%)</div>', unsafe_allow_html=True)
    sector_filtered = sector_df[
        (sector_df['Date'] >= pd.Timestamp(date_range[0])) &
        (sector_df['Date'] <= pd.Timestamp(date_range[1]))
    ]
    fig2 = px.line(
        sector_filtered, x='Date', y=selected_sectors,
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300, legend_title="Sector")
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Correlation heatmap + Distribution ──────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="section-header">🔗 Macro Indicator Correlation Matrix</div>', unsafe_allow_html=True)
    corr = filtered[['GDP_Growth','CPI_Inflation','Repo_Rate','IIP_Growth','INR_USD','Trade_Deficit_USD_bn']].corr()
    fig3 = px.imshow(
        corr, text_auto=True, color_continuous_scale='RdBu_r',
        template='plotly_dark', zmin=-1, zmax=1
    )
    fig3.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=320)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown(f'<div class="section-header">📊 Distribution: {selected_indicator.replace("_"," ")}</div>', unsafe_allow_html=True)
    fig4 = px.histogram(
        filtered, x=selected_indicator, nbins=25,
        color_discrete_sequence=['#4f8bf9'],
        template='plotly_dark', marginal='box'
    )
    fig4.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=320)
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Statistical Analysis ────────────────────────────────────────────────
if show_analysis:
    st.markdown("---")
    st.markdown('<div class="section-header">🧪 Statistical Analysis</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("**ADF Stationarity Test**")
        adf_result = run_stationarity_test(filtered[selected_indicator])
        st.dataframe(pd.DataFrame([adf_result]), use_container_width=True)

        st.markdown("**Descriptive Statistics**")
        desc = filtered[selected_indicator].describe().round(3)
        st.dataframe(desc.to_frame(), use_container_width=True)

    with a2:
        st.markdown("**Time Series Decomposition**")
        if len(filtered) >= 24:
            decomp = run_decomposition(filtered[selected_indicator], filtered['Date'])
            fig5 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  subplot_titles=['Trend', 'Seasonal', 'Residual'])
            fig5.add_trace(go.Scatter(x=decomp['Date'], y=decomp['Trend'], name='Trend',
                                       line=dict(color='#4f8bf9')), row=1, col=1)
            fig5.add_trace(go.Scatter(x=decomp['Date'], y=decomp['Seasonal'], name='Seasonal',
                                       line=dict(color='#f9844a')), row=2, col=1)
            fig5.add_trace(go.Scatter(x=decomp['Date'], y=decomp['Residual'], name='Residual',
                                       line=dict(color='#90e0ef')), row=3, col=1)
            fig5.update_layout(template='plotly_dark', height=380,
                                margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Select at least 24 months for decomposition.")

# ── Row 4: Sector Radar ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🕸️ Sector Performance Radar (Latest 12M Avg)</div>', unsafe_allow_html=True)

recent = sector_df.tail(12)[selected_sectors].mean()
fig6 = go.Figure(go.Scatterpolar(
    r=recent.values,
    theta=recent.index.tolist(),
    fill='toself',
    fillcolor='rgba(79,139,249,0.2)',
    line=dict(color='#4f8bf9')
))
fig6.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    template='plotly_dark',
    height=380,
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")
st.caption("Built by Aarushi K Sharma | MSc Business Statistics, VIT | Research Associate")