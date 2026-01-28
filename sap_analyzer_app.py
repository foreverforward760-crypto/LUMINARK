
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from sap_corporate_analyzer import CorporateSAPAnalyzer, CorporateMetrics, SPATVectors, SAPStage

# Page Configuration
st.set_page_config(
    page_title="LUMINARK SAP Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #FFD700 0%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        color: #FFF;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .trap-alert {
        background-color: #721c24;
        color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .container-alert {
        background-color: #856404;
        color: #fff3cd;
        border: 1px solid #ffeeba;
    }
    .success-box {
        background-color: #155724;
        color: #d4edda;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# sidebar initialization
with st.sidebar:
    st.image("https://placehold.co/200x50/000000/FFF.png?text=LUMINARK", use_container_width=True)
    st.header("Corporate Diagnostics")
    st.markdown("---")
    
    # --- FINANCIAL METRICS ---
    st.subheader("💰 Financial Metrics")
    revenue_growth = st.slider("Revenue Growth (YoY %)", -20.0, 100.0, 3.0, help="Year-over-year revenue growth percentage.")
    profit_margin = st.slider("Profit Margin (%)", -20.0, 60.0, 25.0)
    debt_to_equity = st.slider("Debt to Equity Ratio", 0.0, 5.0, 0.5)
    cash_reserves = st.slider("Cash Reserves (Months)", 0.0, 36.0, 18.0)
    rd_spending = st.slider("R&D Spending (% of Rev)", 0.0, 50.0, 18.0)

    # --- ORGANIZATIONAL METRICS ---
    st.subheader("🏛️ Organizational")
    turnover = st.slider("Employee Turnover (%)", 0.0, 50.0, 22.0)
    leadership_tenure = st.slider("Leadership Tenure (Yrs)", 0.0, 20.0, 8.0)
    management_layers = st.number_input("Management Layers", 1, 20, 12, step=1)
    satisfaction = st.slider("Employee Satisfaction (0-10)", 0.0, 10.0, 6.5)

    # --- MARKET POSITION ---
    st.subheader("🌍 Market Position")
    market_share = st.slider("Market Share (%)", 0.0, 100.0, 35.0)
    industry_growth = st.slider("Industry Growth (%)", -10.0, 50.0, 2.0)
    threats = st.number_input("Competitive Threats", 0, 20, 8)

    # --- INNOVATION & CULTURE ---
    st.subheader("💡 Innovation & Culture")
    new_products = st.number_input("New Products (Last 2 Yrs)", 0, 20, 1)
    innovation_index = st.slider("Innovation Index (0-10)", 0.0, 10.0, 4.5)
    ceo_tenure = st.slider("CEO Tenure (Years)", 0.0, 30.0, 15.0)
    mission_clarity = st.slider("Mission Clarity (0-10)", 0.0, 10.0, 7.0)

    # --- EXTERNAL SIGNALS ---
    st.subheader("📡 External Signals")
    news_sentiment = st.slider("News Sentiment (-1 to 1)", -1.0, 1.0, 0.3)
    analyst_rating = st.selectbox("Analyst Rating", ["buy", "hold", "sell"], index=1)

    run_analysis = st.button("RUN DIAGNOSTIC", type="primary")

# Main Logic
analyzer = CorporateSAPAnalyzer()

# Construct Metrics Object
metrics = CorporateMetrics(
    revenue_growth_yoy=revenue_growth,
    profit_margin=profit_margin,
    debt_to_equity=debt_to_equity,
    cash_reserves_months=cash_reserves,
    r_d_spending_pct=rd_spending,
    employee_turnover_pct=turnover,
    leadership_tenure_years=leadership_tenure,
    management_layers=management_layers,
    employee_satisfaction=satisfaction,
    market_share_pct=market_share,
    market_growth_yoy=industry_growth,
    competitive_threats=threats,
    new_products_last_2yrs=new_products,
    innovation_index=innovation_index,
    ceo_tenure_years=ceo_tenure,
    mission_clarity=mission_clarity,
    news_sentiment=news_sentiment,
    analyst_rating=analyst_rating
)

if run_analysis:
    result = analyzer.analyze_company("Target Entity", metrics)
    
    # --- HEADER ---
    st.markdown(f"<h1 class='main-header'>SAP DIAGNOSTIC REPORT</h1>", unsafe_allow_html=True)
    st.markdown(f"### Target Entity: **{result['company']}**")
    
    # --- TOP LEVEL STAGE ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stage_name = result['stage_name']
        stage_id = result['sap_stage']
        
        # Color coding for stages
        colors = {
            "UNITY_PEAK": "red", "INTEGRATION": "gold", "FOUNDATION": "green", 
            "THRESHOLD": "orange", "CRISIS": "darkred", "PLENARA": "gray"
        }
        color = colors.get(stage_name, "blue")
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white;">
            <h2 style='margin:0'>STAGE {stage_id}: {stage_name}</h2>
            <p style='margin:0'>Confidence: {result['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Investment Signal
        signal = result['investment_signal']
        st.markdown("### 🚦 INVESTMENT SIGNAL")
        st.info(f"**ACTION: {signal['action']}**\n\nReason: {signal['reason']}\n\nRisk: {signal['risk']}")

    with col2:
        # SPAT CHART
        vectors = result['spat_vectors']
        categories = list(vectors.keys())
        values = list(vectors.values())
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=[c.upper() for c in categories],
            fill='toself',
            name='Target'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10])
            ),
            showlegend=False,
            height=300,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- WARNINGS & TRAPS ---
    st.markdown("---")
    st.subheader("🔎 Advanced Diagnostics")

    # Permanence Trap Check
    if "stage_specific_analysis" in result and "permanence_trap" in result["stage_specific_analysis"]:
        trap = result["stage_specific_analysis"]["permanence_trap"]
        if trap["trap_score"] >= 2:
            st.markdown(f"""
            <div class='alert-box trap-alert'>
                ⚠️ PERMANENCE TRAP DETECTED<br>
                Risk Level: {trap['risk_level']}<br>
                {trap['recommendation']}
            </div>
            """, unsafe_allow_html=True)
            for ind in trap["indicators"]:
                st.write(f"- 🔴 {ind}")
        else:
             st.markdown(f"<div class='alert-box success-box'>✅ No Permanence Trap Detected</div>", unsafe_allow_html=True)

    # Container Analysis (if Stage 4) or General Container check if close to 4
    if "stage_specific_analysis" in result and "container_analysis" in result["stage_specific_analysis"]:
        cont = result["stage_specific_analysis"]["container_analysis"]
        st.markdown("#### 📦 Container Rule Analysis")
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Container Ratio", cont['container_ratio'])
        col_c2.metric("Structure Cap", cont['structure_capacity'])
        col_c3.metric("Complexity Load", cont['complexity_load'])
        
        if cont['container_ratio'] < 0.8:
            st.markdown(f"<div class='alert-box container-alert'>⚠️ {cont['status']}</div>", unsafe_allow_html=True)
        else:
            st.success(f"{cont['status']}")

    # --- RAW DATA ---
    with st.expander("View Raw Diagnostic JSON"):
        st.json(result)

else:
    st.info("👈 Adjust metrics in the sidebar and click RUN DIAGNOSTIC")
