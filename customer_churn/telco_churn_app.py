import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_PATH = 'logistic_regression_model.pkl'

# The definitive, ordered list of 29 features provided by your model file.
# This list structure is CRITICAL for scikit-learn deployment.
FEATURE_COLS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 
    'MultipleLines_No phone service', 'MultipleLines_Yes', 
    'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 
    'StreamingTV_No internet service', 'StreamingTV_Yes', 
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 
    'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
]

# --- Helper Functions ---
@st.cache_resource
def load_model(path):
    """Loads the pre-trained model."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please ensure 'logistic_regression_model.pkl' is in the current directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_input_df(user_input):
    """
    Creates the standardized DataFrame for prediction based on user input,
    matching the exact 29 features the model was trained on.
    """
    # 1. Create a dictionary to hold the 29 feature values, initialized to 0
    feature_dict = {col: 0 for col in FEATURE_COLS}
    
    # 2. Set the numerical features directly
    # Note: Gender is completely excluded as it was not used by the model.
    feature_dict['tenure'] = user_input['tenure']
    feature_dict['MonthlyCharges'] = user_input['MonthlyCharges']
    feature_dict['TotalCharges'] = user_input['TotalCharges']
    feature_dict['SeniorCitizen'] = user_input['SeniorCitizen']
    
    # 3. Set the one-hot encoded features (set the corresponding column to 1).
    # We only set the features that EXIST in the FEATURE_COLS list.
    
    # Partner (Reference: 'No')
    if user_input['Partner'] == 'Yes':
        feature_dict['Partner_Yes'] = 1
        
    # Dependents (Reference: 'No')
    if user_input['Dependents'] == 'Yes':
        feature_dict['Dependents_Yes'] = 1

    # Phone Service (Reference: 'No')
    if user_input['PhoneService'] == 'Yes':
        feature_dict['PhoneService_Yes'] = 1

    # Multiple Lines (Reference: 'No')
    if user_input['MultipleLines'] == 'Yes':
        feature_dict['MultipleLines_Yes'] = 1
    elif user_input['MultipleLines'] == 'No phone service':
        feature_dict['MultipleLines_No phone service'] = 1
        
    # Internet Service (Reference: 'DSL')
    if user_input['InternetService'] == 'Fiber optic':
        feature_dict['InternetService_Fiber optic'] = 1
    elif user_input['InternetService'] == 'No':
        feature_dict['InternetService_No'] = 1
        
    # Online Security (Reference: 'No')
    if user_input['OnlineSecurity'] == 'Yes':
        feature_dict['OnlineSecurity_Yes'] = 1
    elif user_input['OnlineSecurity'] == 'No internet service':
        feature_dict['OnlineSecurity_No internet service'] = 1

    # Online Backup (Reference: 'No')
    if user_input['OnlineBackup'] == 'Yes':
        feature_dict['OnlineBackup_Yes'] = 1
    elif user_input['OnlineBackup'] == 'No internet service':
        feature_dict['OnlineBackup_No internet service'] = 1
        
    # Device Protection (Reference: 'No')
    if user_input['DeviceProtection'] == 'Yes':
        feature_dict['DeviceProtection_Yes'] = 1
    elif user_input['DeviceProtection'] == 'No internet service':
        feature_dict['DeviceProtection_No internet service'] = 1

    # Tech Support (Reference: 'No')
    if user_input['TechSupport'] == 'Yes':
        feature_dict['TechSupport_Yes'] = 1
    elif user_input['TechSupport'] == 'No internet service':
        feature_dict['TechSupport_No internet service'] = 1

    # Streaming TV (Reference: 'No')
    if user_input['StreamingTV'] == 'Yes':
        feature_dict['StreamingTV_Yes'] = 1
    elif user_input['StreamingTV'] == 'No internet service':
        feature_dict['StreamingTV_No internet service'] = 1
        
    # Streaming Movies (Reference: 'No')
    if user_input['StreamingMovies'] == 'Yes':
        feature_dict['StreamingMovies_Yes'] = 1
    elif user_input['StreamingMovies'] == 'No internet service':
        feature_dict['StreamingMovies_No internet service'] = 1
        
    # Contract (Reference: 'Month-to-month')
    if user_input['Contract'] == 'One year':
        feature_dict['Contract_One year'] = 1
    elif user_input['Contract'] == 'Two year':
        feature_dict['Contract_Two year'] = 1

    # Paperless Billing (Reference: 'No')
    if user_input['PaperlessBilling'] == 'Yes':
        feature_dict['PaperlessBilling_Yes'] = 1
        
    # Payment Method (Reference: 'Bank transfer (automatic)')
    if user_input['PaymentMethod'] == 'Credit card (automatic)':
        feature_dict['PaymentMethod_Credit card (automatic)'] = 1
    elif user_input['PaymentMethod'] == 'Electronic check':
        feature_dict['PaymentMethod_Electronic check'] = 1
    elif user_input['PaymentMethod'] == 'Mailed check':
        feature_dict['PaymentMethod_Mailed check'] = 1

    # 4. Create the DataFrame in the exact 29-column order the model expects
    input_df = pd.DataFrame([feature_dict])[FEATURE_COLS]
    return input_df

# --- Streamlit App ---

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("📞 Telco Customer Churn Prediction")
st.markdown("Use the sliders and options below to define a customer profile and predict their likelihood of churning.")

# Load the model
model = load_model(MODEL_PATH)

if model:
    # --- Input Fields ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Service Usage")
        
        tenure = st.slider("Tenure (Months)", 1, 72, 24)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 50.0)
        
        # Calculate Total Charges (simplified)
        total_charges = round(tenure * monthly_charges, 2)
        if total_charges > 8684.8: # Cap at max TotalCharges in dataset
            total_charges = 8684.8
        st.metric(label="Calculated Total Charges ($)", value=f"{total_charges:,.2f}")
        
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        payment_method = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        paperless_billing = st.radio("Paperless Billing", ['Yes', 'No'])

    with col2:
        st.header("Account Details")
        
        # Gender is not used by this model, but we still capture it for completeness.
        gender = st.radio("Gender (Not used by model)", ['Male', 'Female'])
        senior_citizen = st.radio("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        partner = st.radio("Partner", ['Yes', 'No'])
        dependents = st.radio("Dependents", ['Yes', 'No'])

        st.subheader("Phone Services")
        phone_service = st.radio("Phone Service", ['Yes', 'No'])
        
        # Only show multiple lines if phone service is 'Yes'
        if phone_service == 'Yes':
            multiple_lines = st.radio("Multiple Lines", ['Yes', 'No'])
        else:
            multiple_lines = 'No phone service' # Matches the OHE column name

    with col3:
        st.header("Internet Services")
        
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        
        # Determine options for internet-related services
        if internet_service == 'No':
            service_options = ['No internet service']
        else:
            service_options = ['Yes', 'No']
            
        online_security = st.selectbox("Online Security", service_options)
        online_backup = st.selectbox("Online Backup", service_options)
        device_protection = st.selectbox("Device Protection", service_options)
        tech_support = st.selectbox("Tech Support", service_options)
        streaming_tv = st.selectbox("Streaming TV", service_options)
        streaming_movies = st.selectbox("Streaming Movies", service_options)
        
    # --- Prediction Logic ---
    st.markdown("---")
    
    # Store all user inputs in a dictionary
    user_input = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': senior_citizen,
        'gender': gender, # Kept for UI but ignored in DataFrame creation
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    # Generate the prediction DataFrame
    input_df = create_input_df(user_input)

    # Perform Prediction
    if st.button("Predict Churn Likelihood", type="primary"):
        try:
            # Predict the probability of churn (class 1)
            prediction_proba = model.predict_proba(input_df)[:, 1][0]
            
            # Convert to percentage
            churn_percentage = prediction_proba * 100
            
            # Determine the message based on probability
            if churn_percentage >= 50:
                st.error(f"High Churn Risk: {churn_percentage:.2f}%")
                st.warning("This customer is predicted to churn. Consider immediate outreach.")
            else:
                st.success(f"Low Churn Risk: {churn_percentage:.2f}%")
                st.info("This customer is predicted to stay.")
                
            st.write(f"The model predicts a **{churn_percentage:.2f}% chance** the customer will churn (leave the company).")

        except Exception as e:
            # Display the critical error message if feature alignment fails
            st.error("An unexpected error occurred during prediction.")
            st.code(f"Error Details: {e}", language='python')
            st.warning("If you still see a feature mismatch error, please double check that you replaced the entire file content correctly.")
