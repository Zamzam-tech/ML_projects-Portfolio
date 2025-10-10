import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# --- Configuration ---
# FIX: Use os.path.dirname(__file__) to correctly find the files 
# relative to the script's location, which resolves the Streamlit Cloud error.
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'demand_forecasting_xgb_model.pkl')
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')

# --- Model Loading and Alignment Functions ---

@st.cache_resource
def load_assets(model_path, feature_path):
    """Loads the trained XGBoost model and the feature blueprint safely."""
    
    # Check if files exist using the corrected, absolute path
    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        st.error("Error: Model or Feature Blueprint not found.")
        st.warning(f"Please run '{os.path.basename(MODEL_PATH)}' first to train the model.")
        return None, None
        
    try:
        model = joblib.load(model_path)
        feature_cols = joblib.load(feature_path)
        return model, feature_cols
    except Exception as e:
        st.error(f"Could not load assets. Error: {e}")
        return None, None

def align_features(user_inputs, feature_cols):
    """
    Converts simple user inputs into the complex, aligned feature vector 
    (DataFrame row) that the XGBoost model requires, including OHE for all 
    categorical columns (Region, Category, Weather Condition, Seasonality, DayName, Month).
    """
    
    # 1. Create a blank DataFrame row with all 0s, using the model's exact column list
    aligned_data = pd.DataFrame(0, index=[0], columns=feature_cols, dtype=float)

    # 2. Map Numerical Features directly
    for key in ['Price', 'Discount', 'Inventory Level']:
        if key in aligned_data.columns:
            aligned_data[key] = user_inputs[key]
        
    # 3. Handle Lagged Features (Simulated Database Lookup)
    if 'Units Sold_Lag_1' in aligned_data.columns:
        aligned_data['Units Sold_Lag_1'] = user_inputs['Units Sold_Lag_1'] 
    if 'Units Sold_Lag_7' in aligned_data.columns:
        aligned_data['Units Sold_7'] = user_inputs['Units Sold_Lag_7']

    # 4. Map One-Hot Encoded (OHE) Features
    
    # Region
    region_col = f'Region_{user_inputs["Region"]}'
    if region_col in aligned_data.columns:
        aligned_data[region_col] = 1.0

    # Category
    category_col = f'Category_{user_inputs["Category"]}'
    if category_col in aligned_data.columns:
        aligned_data[category_col] = 1.0

    # Weather Condition
    weather_col = f'Weather Condition_{user_inputs["Weather Condition"]}'
    if weather_col in aligned_data.columns:
        aligned_data[weather_col] = 1.0

    # Seasonality
    seasonality_col = f'Seasonality_{user_inputs["Seasonality"]}'
    if seasonality_col in aligned_data.columns:
        aligned_data[seasonality_col] = 1.0

    # Day of Week 
    day_of_week_col = f'dayname_{user_inputs["DayName"]}'
    if day_of_week_col in aligned_data.columns:
        aligned_data[day_of_week_col] = 1.0

    # Month 
    month_col = f'month_{user_inputs["Month"]}'
    if month_col in aligned_data.columns:
        aligned_data[month_col] = 1.0
        
    return aligned_data

# --- Streamlit Application ---

def main():
    st.set_page_config(
        page_title="Product Demand Forecast (XGBoost)",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📦 Supply Chain Demand Forecast (XGBoost)")
    st.markdown("Use the controls to set the future conditions and predict the units sold for a product.")
    
    # Load Model and Blueprint
    model, feature_cols = load_assets(MODEL_PATH, FEATURE_COLUMNS_PATH)
    
    if model is None:
        return # Stop execution if assets aren't found

    # User Input Collection (Sidebar)
    st.sidebar.header("Input Features")
    
    # 1. Core Variables
    st.sidebar.subheader("Market & Product Data")
    
    input_date = st.sidebar.date_input(
        "Forecast Date",
        datetime.now().date(),
        help="The date determines the Day of Week and Month features."
    )
    
    price = st.sidebar.number_input("Price ($)", min_value=1.0, max_value=100.0, value=25.0, step=0.5, format="%.2f")
    discount = st.sidebar.number_input("Discount (%)", min_value=0, max_value=50, value=10, help="e.g., 10 for a 10% discount.")
    inventory = st.sidebar.number_input("Inventory Level (Units)", min_value=0, value=500, step=10)

    # 2. Categorical Variables (Must match the OHE names in the training script)
    st.sidebar.subheader("Categorical Drivers")
    colA, colB = st.sidebar.columns(2)
    with colA:
        region = st.selectbox("Region", ['East', 'West', 'North', 'South'], index=0)
    with colB:
        category = st.selectbox("Product Category", ['A', 'B', 'C'], index=0)
        
    colC, colD = st.sidebar.columns(2)
    with colC:
        # NEW INPUT
        weather = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy'], index=0)
    with colD:
        # NEW INPUT
        seasonality = st.selectbox("Seasonality (Quarter)", ['Q1', 'Q2', 'Q3', 'Q4'], index=0)

    # 3. Lagged Feature Simulation (Crucial for Time Series)
    st.sidebar.subheader("Lagged Sales (Historical Data)")
    st.sidebar.markdown("_These represent yesterday's and last week's sales figures._")
    lag_1 = st.sidebar.number_input("Yesterday's Sales (Lag 1)", value=45, min_value=0, step=1)
    lag_7 = st.sidebar.number_input("Last Week's Sales (Lag 7)", value=120, min_value=0, step=1)

    # --- Prepare Data for Prediction ---

    # Extract features from date input
    day_name = input_date.strftime('%A')
    month = input_date.month

    user_inputs = {
        'Price': price,
        'Discount': discount / 100.0, # Convert % to decimal
        'Inventory Level': inventory, 
        'Region': region,
        'Category': category,
        'Weather Condition': weather, # Added to inputs
        'Seasonality': seasonality,   # Added to inputs
        'Units Sold_Lag_1': float(lag_1),
        'Units Sold_Lag_7': float(lag_7),
        'DayName': day_name,
        'Month': month
    }
    
    # 4. Align the data for the model
    aligned_data = align_features(user_inputs, feature_cols)

    # --- Prediction Logic ---
    st.markdown("---")
    if st.button("Calculate Forecast", type="primary"):
        # Make the prediction
        prediction = model.predict(aligned_data)[0]
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Units Sold", 
                value=f"{max(0, int(prediction)):,}", 
                delta_color="off"
            )
            st.success("Forecast generated successfully!")
        
        with col2:
            st.info(f"Forecast for: **{input_date.strftime('%A, %B %d, %Y')}**")
            st.markdown(f"""
            **Key Drivers:**
            - Price: ${price:.2f} (Discount: {discount}%)
            - Last Week's Sales (Lag 7): {lag_7} units
            - Weather: {weather} | Seasonality: {seasonality}
            """)
            
    else:
        st.info("Adjust the parameters in the sidebar and click 'Calculate Forecast' to get started.")

if __name__ == "__main__":
    main()


