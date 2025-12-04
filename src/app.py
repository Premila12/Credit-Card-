import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os
import base64
import io 

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_data
from utils.risk_engine import calculate_risk_scores, get_active_model_info
from ml_pipeline.data_manager import DataManager

# --- Configuration ---
st.set_page_config(
    page_title="HDFC Early Risk Monitor",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

# --- Helper: Load Image as Base64 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Validate CSV Format ---
def validate_csv_format(df):
    """Check if CSV has required columns"""
    required_cols = [
        'customer_id', 
        'utilisation_pct', 
        'avg_payment_ratio',
        'cash_withdrawal_pct', 
        'recent_spend_change_pct'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, required_cols, missing
    return True, required_cols, []

# --- Load CSS ---
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Header (Pure HTML for perfect centering) ---
logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(logo_path):
    logo_base64 = get_base64_of_bin_file(logo_path)
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="logo-img">'
else:
    logo_html = "<h1 style='text-align: center; color: #0057A5;'>HDFC BANK</h1>"

st.markdown(
    f"""
    <div class="logo-container">
        {logo_html}
        <div class="welcome-text">Credit Card Delinquency Watch</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar (Data Upload) ---
with st.sidebar:
    st.title("🛡️ Risk Watch")
    st.markdown("Upload your customer profile data to detect early delinquency signals.")
    
    uploaded_file = st.file_uploader("Upload Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        st.success("File Uploaded Successfully!")
        # Validate format immediately after upload
        try:
            temp_df = pd.read_csv(uploaded_file)
            is_valid, required_cols, missing_cols = validate_csv_format(temp_df)
            
            if not is_valid:
                st.error(f"❌ Invalid CSV Format!")
                st.error(f"Missing columns: `{', '.join(missing_cols)}`")
                with st.expander("📋 Required Columns"):
                    st.markdown("""
                    Your CSV must contain these columns:
                    - `customer_id`: Unique identifier (e.g., C001, C002)
                    - `utilisation_pct`: Credit utilization % (0-100)
                    - `avg_payment_ratio`: Average payment ratio % (0-100)
                    - `cash_withdrawal_pct`: Cash withdrawal % (0-100)
                    - `recent_spend_change_pct`: Recent spend change % (-100 to 100)
                    
                    **Example row:**
                    | customer_id | utilisation_pct | avg_payment_ratio | cash_withdrawal_pct | recent_spend_change_pct |
                    |---|---|---|---|---|
                    | C001 | 45 | 85 | 15 | 10 |
                    """)
                uploaded_file = None  # Reset to use sample data
            else:
                # Reset file pointer before saving (file was already read for validation)
                uploaded_file.seek(0)
                # Save uploaded file for future training
                try:
                    data_manager = DataManager()
                    data_manager.store_new_data(uploaded_file, uploaded_file.name)
                    st.caption("📊 Data saved for model improvement")
                except Exception as e:
                    st.warning(f"Could not save file: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            uploaded_file = None
    else:
        st.info("Loading sample data automatically...")
    
    # Display active model info
    st.markdown("---")
    try:
        model_info = get_active_model_info()
        st.caption(f"**Model:** v{model_info['version']}")
        if model_info['date'] != 'N/A':
            st.caption(f"**Updated:** {model_info['date'][:10]}")
    except:
        pass

# --- Data Loading with Caching ---
@st.cache_data
def process_data(data_source):
    """Load and process data with error handling"""
    try:
        # Load data
        df = load_data(data_source)
        
        if df is None:
            return None
        
        # Validate format
        is_valid, required_cols, missing_cols = validate_csv_format(df)
        if not is_valid:
            st.error(f" Data validation failed!")
            st.error(f"Missing columns: `{', '.join(missing_cols)}`")
            return None
        
        # Calculate risk scores
        return calculate_risk_scores(df)
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Load data (prioritize uploaded file, fallback to sample)
if uploaded_file is not None:
    df = process_data(uploaded_file)
else:
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_data.csv")
    if os.path.exists(sample_path):
        df = process_data(sample_path)
    else:
        st.error("Sample data not found. Please upload a CSV file.")
        st.stop()

if df is None:
    st.error("Failed to process data. Please check file format and try again.")
    st.stop()

# --- Search Logic ---
c_fill, c_search = st.columns([5, 2])
with c_search:
    search_query = st.text_input("Search Customer ID", placeholder="e.g., C005", label_visibility="collapsed")

if search_query:
    search_query = search_query.strip().upper()
    if search_query in df['customer_id'].values:
        customer_data = df[df['customer_id'] == search_query].iloc[0]
        risk_tier = customer_data['risk_tier']
        risk_score = customer_data['risk_score']
        
        # Color-coded success message based on risk tier
        if risk_tier == 'Intervene':
            st.error(f"⚠️ Customer {search_query} found - HIGH RISK ({risk_tier}) | Score: {risk_score}")
        elif risk_tier == 'Engage':
            st.warning(f"⚠️ Customer {search_query} found - MEDIUM RISK ({risk_tier}) | Score: {risk_score}")
        else:
            st.success(f"✅ Customer {search_query} found - LOW RISK ({risk_tier}) | Score: {risk_score}")
        
        df = df[df['customer_id'] == search_query]
    else:
        st.warning(f"Customer ID '{search_query}' not found.")

st.markdown("---")

# --- Navigation ---
tabs = st.tabs(["Dashboard", "Customer Investigation", "High Risk Alerts", "Drilldown"])

# --- Tab 1: Dashboard (Visuals) ---
with tabs[0]:
    st.markdown("### Portfolio Risk Overview")
    
    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", len(df))
    m2.metric("High Risk", len(df[df['risk_tier'] == 'Intervene']), delta_color="inverse")
    m3.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")
    m4.metric("Avg Utilization", f"{df['utilisation_pct'].mean():.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Risk Distribution")
        fig_pie = px.pie(df, names='risk_tier', title='Portfolio Risk Tiers', 
                         color='risk_tier',
                         color_discrete_map={'Intervene':'#E31837', 'Engage':'#FFA500', 'Monitor':'#0057A5'},
                         hole=0.4)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="black")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("Risk Drivers: Utilization vs Score")
        fig_scatter = px.scatter(df, x='utilisation_pct', y='risk_score', 
                                 color='risk_tier', hover_data=['customer_id'],
                                 title='Higher Utilization Correlates with Risk',
                                 color_discrete_map={'Intervene':'#E31837', 'Engage':'#FFA500', 'Monitor':'#0057A5'})
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="black")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Tab 2: Customer Investigation (Detailed View) ---
with tabs[1]:
    st.markdown("### Customer Investigation")
    
    if not search_query:
        selected_customer = st.selectbox("Select Customer", df['customer_id'].unique())
    else:
        selected_customer = df['customer_id'].iloc[0]
    
    if selected_customer:
        if not df[df['customer_id'] == selected_customer].empty:
            cust = df[df['customer_id'] == selected_customer].iloc[0]
            
            # Status Banner
            color = "#E31837" if cust['risk_tier'] == 'Intervene' else "#FFA500" if cust['risk_tier'] == 'Engage' else "#0057A5"
            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 8px; color: white; font-weight: bold; margin-bottom: 20px;">
                Customer {selected_customer} is {cust['risk_tier'].upper()} (Score: {cust['risk_score']})
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Tiles
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Utilization", f"{cust['utilisation_pct']}%")
            m2.metric("Avg Payment Ratio", f"{cust['avg_payment_ratio']}%")
            m3.metric("Cash Withdrawal", f"{cust['cash_withdrawal_pct']}%")
            m4.metric("Spend Change", f"{cust['recent_spend_change_pct']}%")
            
            st.markdown("#### Full Profile")
            st.table(pd.DataFrame([cust]))
            
            # Radar Chart
            st.subheader("Risk Radar")
            categories = ['Utilization', 'Payment Gap', 'Cash', 'Trend']
            values = [
                min(cust['utilisation_pct'], 100), 
                max(0, 100 - cust['avg_payment_ratio']),
                min(cust['cash_withdrawal_pct'] * 2, 100),
                max(0, cust['recent_spend_change_pct'])
            ]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Customer Risk',
                line_color=color
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="black", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: High Risk Alerts ---
with tabs[2]:
    st.markdown("###  High-Risk Accounts Detected 🚨")
    high_risk_df = df[df['risk_tier'] == 'Intervene'].copy()
    if not high_risk_df.empty:
        high_risk_df_display = high_risk_df.reset_index(drop=True)
        high_risk_df_display.insert(0, 'Sl No', range(1, len(high_risk_df_display) + 1))
        st.write(high_risk_df_display.to_html(index=False), unsafe_allow_html=True)
    else:
        st.success("No High-Risk accounts detected.")

# --- Tab 4: Drilldown ---
with tabs[3]:
    st.markdown("### Customer Drill-Down")
    
    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        tier_filter = st.multiselect("Filter by Tier", ['Intervene', 'Engage', 'Monitor'], default=['Intervene', 'Engage', 'Monitor'])
    with c2:
        score_range = st.slider("Score Range", 0.0, 100.0, (0.0, 100.0))
    with c3:
        top_n = st.number_input("Top N Customers", min_value=5, max_value=100, value=100)
        
    # Apply Filters
    filtered_df = df.copy()
    if tier_filter:
        filtered_df = filtered_df[filtered_df['risk_tier'].isin(tier_filter)]
    
    filtered_df = filtered_df[
        (filtered_df['risk_score'] >= score_range[0]) & 
        (filtered_df['risk_score'] <= score_range[1])
    ]
    
    filtered_df = filtered_df.sort_values('risk_score', ascending=False).head(top_n)
    
    if not filtered_df.empty:
        filtered_df_display = filtered_df.reset_index(drop=True)
        filtered_df_display.insert(0, 'Sl No', range(1, len(filtered_df_display) + 1))
        st.write(filtered_df_display.to_html(index=False), unsafe_allow_html=True)
    else:
        st.warning("No customers match the selected filters.")
