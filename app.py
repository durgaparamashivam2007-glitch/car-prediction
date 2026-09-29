import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from pathlib import Path

from train_model import (
    load_and_preprocess_raw,
    train_and_save_pipeline,
    BASE_DIR,
    PIPELINE_PATH,
    METADATA_PATH,
    DATASET_PATH,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL
)

# Page Configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Card */
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Metric Cards */
    .metric-container {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    /* Prediction Output Card */
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.12) 100%);
        border: 2px solid #10b981;
        border-radius: 18px;
        padding: 26px;
        text-align: center;
        margin-top: 20px;
    }
    
    .result-price {
        font-size: 2.8rem;
        font-weight: 800;
        color: #10b981;
        margin: 10px 0;
    }

    .result-words {
        font-size: 1.25rem;
        font-weight: 600;
        color: #6ee7b7;
    }

    /* Info Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Form input fields override */
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format INR currency
def format_inr(number):
    val = round(number)
    s = f"{val:,.0f}"
    if val >= 10000000:
        words = f"₹ {val / 10000000:.2f} Crore"
    elif val >= 100000:
        words = f"₹ {val / 100000:.2f} Lakhs"
    else:
        words = f"₹ {val:,.0f}"
    return f"₹ {s}", words

# Cached Data & Model Loaders
@st.cache_resource
def load_pipeline():
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        with st.spinner("Initializing ML Pipeline for the first time..."):
            pipeline, metadata = train_and_save_pipeline()
            return pipeline, metadata
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        metadata = joblib.load(METADATA_PATH)
        return pipeline, metadata
    except Exception as e:
        st.warning("Re-training model pipeline due to loading notice...")
        return train_and_save_pipeline()

@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        st.error(f"Missing required dataset: `{DATASET_PATH}`")
        st.stop()
    df, raw_count, cleaned_count, duplicates_removed = load_and_preprocess_raw(DATASET_PATH)
    return df, raw_count, cleaned_count, duplicates_removed

# Load data and pipeline safely
try:
    pipeline, metadata = load_pipeline()
    df_clean, raw_count, cleaned_count, duplicates_removed = load_dataset()
except Exception as err:
    st.error(f"Application Error: {err}")
    st.stop()

# Manage Session State for Navigation
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "🏠 Home"

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
menu_options = [
    "🏠 Home",
    "🚗 Price Prediction",
    "📊 Dataset Insights",
    "🤖 Model Performance",
    "ℹ️ About"
]

# Sync radio selection with session state
selected_page = st.sidebar.radio(
    "Go to:",
    options=menu_options,
    index=menu_options.index(st.session_state["nav_selection"])
)
st.session_state["nav_selection"] = selected_page

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: rgba(30, 41, 59, 0.7); padding: 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
    <strong style="color: #38bdf8;">Machine Learning Project</strong><br>
    <span style="font-size: 0.88rem; color: #cbd5e1;">Car Price Prediction</span><br>
    <small style="color: #94a3b8;">Powered by Python + Scikit-learn + Streamlit</small>
</div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: HOME
# ==========================================
if selected_page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🚗 CAR PRICE PREDICTION</div>
        <div class="hero-subtitle">Estimate the market value of a used car using machine learning.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📌 Project Highlights")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{cleaned_count:,}</div>
            <div class="metric-label">📊 Dataset Records</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">11</div>
            <div class="metric-label">🔢 Features</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">6</div>
            <div class="metric-label">🤖 ML Models</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        r2_val = metadata['results'][metadata['best_model_name']]['R2']
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{r2_val*100:.1f}%</div>
            <div class="metric-label">🎯 Model R²</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("The application analyzes vehicle specifications and historical pricing data from `car details v3.csv` to estimate the selling price of a used car.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start Price Prediction →", use_container_width=True):
        st.session_state["nav_selection"] = "🚗 Price Prediction"
        st.rerun()


# ==========================================
# PAGE 2: PRICE PREDICTION
# ==========================================
elif selected_page == "🚗 Price Prediction":
    st.title("🚗 Car Price Predictor")
    st.caption("AI-powered used car price estimation")
    
    st.markdown("---")
    
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚘 Vehicle Information")
            brand = st.selectbox(
                "Brand / Car Name",
                options=metadata['unique_brands'],
                index=metadata['unique_brands'].index("Maruti") if "Maruti" in metadata['unique_brands'] else 0
            )
            year = st.number_input(
                "Manufacturing Year",
                min_value=1990,
                max_value=2026,
                value=2017,
                step=1
            )
            fuel = st.selectbox(
                "Fuel Type",
                options=metadata['unique_fuels'],
                index=0
            )

            st.markdown("### 💰 Seller & Owner Information")
            seller_type = st.selectbox(
                "Seller Type",
                options=metadata['unique_seller_types'],
                index=0
            )
            transmission = st.selectbox(
                "Transmission",
                options=metadata['unique_transmissions'],
                index=0
            )
            owner = st.selectbox(
                "Owner Information",
                options=metadata['unique_owners'],
                index=0
            )

        with col2:
            st.markdown("### ⚙️ Vehicle Specifications")
            km_driven = st.number_input(
                "Kilometers Driven (km)",
                min_value=100,
                max_value=1000000,
                value=50000,
                step=2500
            )
            mileage = st.number_input(
                "Fuel Mileage (kmpl / km/kg)",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=0.5
            )
            engine = st.number_input(
                "Engine Displacement (CC)",
                min_value=500,
                max_value=5000,
                value=1248,
                step=50
            )
            max_power = st.number_input(
                "Max Engine Power (bhp)",
                min_value=20.0,
                max_value=600.0,
                value=85.0,
                step=1.0
            )
            seats = st.selectbox(
                "Seating Capacity",
                options=[2, 4, 5, 6, 7, 8, 9, 10, 14],
                index=2
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔮 Predict Car Price")

    if submit_btn:
        input_data = pd.DataFrame([{
            'brand': brand,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'year': int(year),
            'km_driven': int(km_driven),
            'mileage_kmpl': float(mileage),
            'engine_cc': float(engine),
            'max_power_bhp': float(max_power),
            'seats': float(seats)
        }])
        
        try:
            pred_val = pipeline.predict(input_data)[0]
            pred_val = max(15000.0, float(pred_val))
            
            formatted_price, word_price = format_inr(pred_val)
            best_model_name = metadata['best_model_name']
            r2_score_val = metadata['results'][best_model_name]['R2']
            
            st.markdown(f"""
            <div class="result-card">
                <div style="color: #94a3b8; font-size: 1rem; font-weight: 600; text-transform: uppercase;">ESTIMATED CAR PRICE</div>
                <div class="result-price">{formatted_price}</div>
                <div class="result-words">({word_price})</div>
                <div style="margin-top: 14px; color: #cbd5e1; font-size: 0.9rem;">
                    Estimated using the trained <b>{best_model_name}</b> machine learning model (Model R² = {r2_score_val*100:.2f}%).
                </div>
                <div style="margin-top: 8px; color: #94a3b8; font-size: 0.8rem; font-style: italic;">
                    * This is an ML-based estimate and actual market prices may vary.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating prediction: {e}")


# ==========================================
# PAGE 3: DATASET INSIGHTS
# ==========================================
elif selected_page == "📊 Dataset Insights":
    st.title("📊 Dataset Insights & EDA")
    st.caption("Visualizing dataset patterns and vehicle price distributions from `car details v3.csv`.")
    
    st.markdown("---")
    
    # Overview metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Total Raw Records", f"{raw_count:,}")
    with col_b:
        st.metric("Cleaned Records", f"{cleaned_count:,}")
    with col_c:
        st.metric("Duplicates Removed", f"{duplicates_removed:,}")
    with col_d:
        st.metric("Total Features", "11")

    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_graphs, tab_summary = st.tabs(["📈 Exploratory Charts", "📋 Numerical Summary"])
    
    with tab_graphs:
        c1, c2 = st.columns(2)
        
        with c1:
            # 1. Price Distribution
            fig_dist = px.histogram(
                df_clean,
                x='selling_price',
                nbins=50,
                title="1. Car Selling Price Distribution (₹)",
                color_discrete_sequence=['#38bdf8']
            )
            fig_dist.update_layout(template='plotly_dark', height=380, xaxis_title="Selling Price (₹)", yaxis_title="Count")
            st.plotly_chart(fig_dist, use_container_width=True)

            # 3. Price vs Kilometers Driven
            fig_km = px.scatter(
                df_clean,
                x='km_driven',
                y='selling_price',
                color='fuel',
                title="3. Price vs Kilometers Driven",
                opacity=0.6,
                labels={'km_driven': 'Kilometers Driven', 'selling_price': 'Price (₹)'}
            )
            fig_km.update_layout(template='plotly_dark', height=380)
            st.plotly_chart(fig_km, use_container_width=True)

            # 5. Price by Transmission
            fig_trans = px.box(
                df_clean,
                x='transmission',
                y='selling_price',
                color='transmission',
                title="5. Price Distribution by Transmission",
                labels={'selling_price': 'Price (₹)', 'transmission': 'Transmission'}
            )
            fig_trans.update_layout(template='plotly_dark', height=380)
            st.plotly_chart(fig_trans, use_container_width=True)

        with c2:
            # 2. Price vs Year
            fig_year = px.box(
                df_clean,
                x='year',
                y='selling_price',
                title="2. Price Distribution by Manufacturing Year",
                color_discrete_sequence=['#818cf8']
            )
            fig_year.update_layout(template='plotly_dark', height=380, xaxis_title="Year", yaxis_title="Price (₹)")
            st.plotly_chart(fig_year, use_container_width=True)

            # 4. Price by Fuel Type
            fig_fuel = px.box(
                df_clean,
                x='fuel',
                y='selling_price',
                color='fuel',
                title="4. Price Distribution by Fuel Type",
                labels={'selling_price': 'Price (₹)', 'fuel': 'Fuel Type'}
            )
            fig_fuel.update_layout(template='plotly_dark', height=380)
            st.plotly_chart(fig_fuel, use_container_width=True)

            # 6. Correlation Heatmap
            num_df = df_clean[['selling_price', 'year', 'km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']].dropna()
            corr = num_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="6. Feature Correlation Heatmap"
            )
            fig_corr.update_layout(template='plotly_dark', height=380)
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab_summary:
        st.subheader("Cleaned Dataset Statistics")
        st.dataframe(df_clean[['selling_price', 'year', 'km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']].describe().T, use_container_width=True)


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
elif selected_page == "🤖 Model Performance":
    st.title("🤖 Model Performance Benchmark")
    st.caption("Evaluation and selection of machine learning algorithms based on MAE, RMSE, and R² Score.")
    
    st.markdown("---")
    
    results = metadata['results']
    best_name = metadata['best_model_name']
    
    st.success(f"🏆 **Selected Model:** **{best_name}** (Achieved highest validation R² Score of **{results[best_name]['R2']:.4f}**)")
    
    # Benchmark table
    res_df = pd.DataFrame(results).T
    res_df['MAE (₹)'] = res_df['MAE'].apply(lambda x: f"₹ {x:,.2f}")
    res_df['RMSE (₹)'] = res_df['RMSE'].apply(lambda x: f"₹ {x:,.2f}")
    res_df['R² Score'] = res_df['R2'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(res_df[['R² Score', 'MAE (₹)', 'RMSE (₹)']], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Chart 1: R2 comparison
        res_raw = pd.DataFrame(results).T.reset_index()
        res_raw.columns = ['Model', 'MAE', 'RMSE', 'R2']
        fig_r2 = px.bar(
            res_raw,
            x='R2',
            y='Model',
            orientation='h',
            title="Model Comparison: R² Score (Higher is Better)",
            color='R2',
            color_continuous_scale='Viridis',
            text_auto='.3f'
        )
        fig_r2.update_layout(template='plotly_dark', height=400, xaxis_title="R² Score", yaxis_title="")
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        # Chart 2: MAE comparison
        fig_mae = px.bar(
            res_raw,
            x='MAE',
            y='Model',
            orientation='h',
            title="Model Comparison: MAE in ₹ (Lower is Better)",
            color='MAE',
            color_continuous_scale='Reds_r',
            text_auto=',.0f'
        )
        fig_mae.update_layout(template='plotly_dark', height=400, xaxis_title="Mean Absolute Error (₹)", yaxis_title="")
        st.plotly_chart(fig_mae, use_container_width=True)


# ==========================================
# PAGE 5: ABOUT
# ==========================================
elif selected_page == "ℹ️ About":
    st.title("ℹ️ About the Project")
    st.caption("College Machine Learning Project Demonstration")
    
    st.markdown("---")
    
    st.markdown("### 📌 Problem Statement")
    st.info(
        "Used car prices depend on multiple factors such as age, mileage, fuel type, transmission and vehicle specifications. "
        "This project uses machine learning to estimate the selling price of a used car based on historical vehicle transaction records."
    )
    
    st.markdown("### 🔄 Machine Learning Workflow")
    st.code(
        "Dataset (car details v3.csv)\n"
        "   ↓\n"
        "Data Cleaning & Feature Extraction (Regex parsing for kmpl, CC, bhp, brand)\n"
        "   ↓\n"
        "Preprocessing (Median Imputation, One-Hot Encoding, Standard Scaling)\n"
        "   ↓\n"
        "Model Training (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)\n"
        "   ↓\n"
        "Model Evaluation (MAE, RMSE, R² Score)\n"
        "   ↓\n"
        "Price Prediction (Streamlit Web Dashboard)",
        language="text"
    )
    
    st.markdown("### 🛠️ Technologies Used")
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("""
        - 🐍 **Python 3.10+**
        - 🐼 **Pandas** (Data cleaning & manipulation)
        - 🔢 **NumPy** (Numerical computing)
        - ⚙️ **Scikit-learn** (Machine learning pipeline & models)
        """)
    with tech_col2:
        st.markdown("""
        - 📊 **Plotly** (Interactive data visualizations)
        - 🎈 **Streamlit** (Web application dashboard)
        - 📦 **Joblib** (Pipeline serialization)
        """)
