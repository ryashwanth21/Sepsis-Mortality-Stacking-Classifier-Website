import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import plotly.graph_objects as go
import plotly.express as px

limits = {
    "max_age": {"min": 19, "max": 101, "mean": 64.78},
    "los_icu": {"min": 1.04, "max": 101.75, "mean": 6.67},
    "sofa_score": {"min": 2, "max": 20, "mean": 4.41},
    "avg_urineoutput": {"min": -100.0, "max": 1537.5, "mean": 173.88},
    "glucose_min": {"min": 3.0, "max": 594.0, "mean": 100.76},
    "glucose_max": {"min": 35.0, "max": 5840.0, "mean": 227.38},
    "glucose_average": {"min": 34.0, "max": 891.0, "mean": 150.63},
    "sodium_min": {"min": 98.0, "max": 167.0, "mean": 133.73},
    "sodium_max": {"min": 113.0, "max": 180.0, "mean": 141.89},
    "sodium_average": {"min": 110.5, "max": 168.0, "mean": 137.91},
    "heart_rate_min": {"min": 1, "max": 134, "mean": 67.36},
    "heart_rate_max": {"min": 42, "max": 257, "mean": 117.37},
    "heart_rate_mean": {"min": 36.0, "max": 151.96, "mean": 88.64},
    "sbp_min": {"min": 1.0, "max": 162.0, "mean": 80.68},
    "sbp_max": {"min": 81.0, "max": 365.0, "mean": 158.16},
    "sbp_mean": {"min": 69.39, "max": 186.42, "mean": 115.7},
    "dbp_min": {"min": 1.0, "max": 86.0, "mean": 40.21},
    "dbp_max": {"min": 44.0, "max": 291.0, "mean": 98.53},
    "dbp_mean": {"min": 33.43, "max": 110.21, "mean": 62.27},
    "resp_rate_min": {"min": 1.0, "max": 30.0, "mean": 10.92},
    "resp_rate_max": {"min": 14.0, "max": 69.0, "mean": 32.48},
    "resp_rate_mean": {"min": 9.88, "max": 38.08, "mean": 19.97},
    "spo2_min": {"min": 1, "max": 100, "mean": 88.1},
    "spo2_max": {"min": 84, "max": 100, "mean": 99.8},
    "spo2_mean": {"min": 73.12, "max": 100.0, "mean": 96.89},
    "albumin": {"min": 1.0, "max": 6.0, "mean": 3.37}
}

# Set page configuration
st.set_page_config(
    page_title="Sepsis Mortality Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and animations
def load_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f8f9fa;
        }
        .h3 {
            color:black !important;
                }
        /* Custom card effect */
        .css-1r6slb0 {
            border-radius: 10px;
            box-shadow: 0 6px 10px rgba(0,0,0,0.08), 0 0 6px rgba(0,0,0,0.05);
            transition: transform .3s;
        }
        .css-1r6slb0:hover {
            transform: scale(1.01);
        }
        
        /* Header styling */
        .title {
            font-size: 3rem;
            font-weight: 700;
            color: #0066cc;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            animation: fadeIn 1.5s;
        }
        
        /* Subtitle styling */
        .subtitle {
            font-size: 1.5rem;
            color: #5c5c5c;
            text-align: center;
            margin-bottom: 2rem;
            animation: slideIn 1.5s;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #303030;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #0066cc;
            animation: fadeIn 2s;
        }
        
        /* Input fields styling */
        div[data-baseweb="input"] input, div[data-baseweb="select"] {
            border-radius: 8px !important;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 20px;
            padding: 10px 24px;
            font-weight: 600;
            background-color: #0066cc;
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        .stVerticalBlock.st-emotion-cache-6wqzdz.eu6p4el3 {
                color:black;
                }
        .stButton>button:hover {
            background-color: #004c99;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Example button styling */
        .example-button button {
            background-color: #28a745;
            margin-right: 10px;
        }
        
        .example-button button:hover {
            background-color: #218838;
        }
        
        /* Results container */
        .results-container {
            padding: 20px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 2rem;
            animation: fadeIn 1s;
        }
        
        /* Prediction result */
        .prediction-result {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            animation: pulse 2s infinite;
        }
        
        .high-risk {
            background-color: rgba(255, 0, 0, 0.1);
            color: #d62728;
            border: 2px solid #d62728;
        }
        
        .low-risk {
            background-color: rgba(0, 128, 0, 0.1);
            color: #2ca02c;
            border: 2px solid #2ca02c;
        }
        
        /* Animations */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        
        @keyframes slideIn {
            0% { transform: translateY(20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 102, 204, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(0, 102, 204, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 102, 204, 0); }
        }
        
        /* Loading animation */
        .loading-animation {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
        }
        
        /* Gauge meter */
        .gauge-container {
            display: flex;
            justify-content: center;
            margin-top: 1rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            color: #666;
            font-size: 0.8rem;
        }
        
        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_css()

# Helper function to get value from session state or use default
def get_value(key, default):
    return st.session_state.form_values.get(key, default)

# Define explicit types
field_types = {
    "max_age": "int",
    "los_icu": "float",
    "sofa_score": "int",
    "avg_urineoutput": "float",
    "glucose_min": "float",
    "glucose_max": "float",
    "glucose_average": "float",
    "sodium_max": "float",
    "sodium_min": "float",
    "sodium_average": "float",
    "heart_rate_min": "int",
    "heart_rate_max": "int",
    "heart_rate_mean": "float",
    "sbp_min": "float",
    "sbp_max": "float",
    "sbp_mean": "float",
    "dbp_min": "float",
    "dbp_max": "float",
    "dbp_mean": "float",
    "resp_rate_min": "float",
    "resp_rate_max": "float",
    "resp_rate_mean": "float",
    "spo2_min": "int",
    "spo2_max": "int",
    "spo2_mean": "float",
    "albumin": "float"
}

# Corrected dynamic input
def dynamic_input(label, field, step=1.0):
    field_min = limits[field]["min"]
    field_max = limits[field]["max"]
    field_mean = limits[field]["mean"]
    default = get_value(field, field_mean)
    
    field_type = field_types.get(field, "float")
    
    if field_type == "int":
        return st.number_input(label, int(field_min), int(field_max), int(default), step=1)
    elif field_type == "float":
        return st.number_input(label, float(field_min), float(field_max), float(default), step=step)
    else:
        return st.number_input(label, field_min, field_max, default, step=step)


# Function to create animation for page loading
def load_animation():
    with st.spinner("Loading dashboard..."):
        time.sleep(1)

# Create header with animation
def create_header():
    st.markdown('<div class="title">ICU Sepsis Mortality Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict patient mortality risk using advanced machine learning</div>', unsafe_allow_html=True)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('Stacking_best_model1.pkl', 'rb'))
        scaler = pickle.load(open('StandardScaler_model2.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model files are in the correct location.")
        return None, None

# Create a background image or gradient
def add_bg_from_gradient():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to create a feature importance plot
def plot_feature_importance(model, feature_names):
    # This function depends on your model type, here's an example using RandomForest base model
    try:
        # Get feature importance from the RandomForest base model in the stacking classifier
        rf_model = model.estimators_[0][1]  # Assuming 'rf' is the first estimator
        importances = rf_model.feature_importances_
        
        # Create dataframe for plotting
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)
        
        # Plot with Plotly
        fig = px.bar(
            feature_imp, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        
        return fig
    except:
        # Fallback if feature importance extraction doesn't work
        st.warning("Feature importance visualization is not available for this model.")
        return None

# Create gauge chart for prediction probability
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Mortality Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': 'green'},
                {'range': [33, 66], 'color': 'yellow'},
                {'range': [66, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#404040", 'family': "Arial"}
    )
    
    return fig

# Define example patient data
def get_example_data():
    # Define three example patient profiles
    example1 = {
        'max_age': 58,
        'los_icu': 2.33,
        'sofa_score': 8,
        'avg_urineoutput': 34.26,
        'glucose_min': 94,
        'glucose_max': 159,
        'glucose_average': 117,
        'sodium_max': 146,
        'sodium_min': 134,
        'sodium_average': 139.625,
        'diabetes_without_cc': 0,
        'diabetes_with_cc': 0,
        'severe_liver_disease': 0,
        'aids': 0,
        'renal_disease': 1,
        'heart_rate_min': 69,
        'heart_rate_max': 130,
        'heart_rate_mean': 89.56,
        'sbp_min': 89,
        'sbp_max': 189,
        'sbp_mean': 129.02,
        'dbp_min': 57,
        'dbp_max': 104,
        'dbp_mean': 78.24,
        'resp_rate_min': 10,
        'resp_rate_max': 47,
        'resp_rate_mean': 22.23,
        'spo2_min': 92,
        'spo2_max': 100,
        'spo2_mean': 96.56,
        'coma': 0,
        'albumin': 3.9,
        'race': "Black or African American",
        'antibiotic_vancomycin': "Yes",
        'gender': "Male"
    }
    
    example2 = {
        'max_age': 65,
        'los_icu': 1.83,
        'sofa_score': 2,
        'avg_urineoutput': 105.48,
        'glucose_min': 65,
        'glucose_max': 95,
        'glucose_average': 81.5,
        'sodium_max': 149,
        'sodium_min': 127,
        'sodium_average': 139.67,
        'diabetes_without_cc': 0,
        'diabetes_with_cc': 0,
        'severe_liver_disease': 0,
        'aids': 0,
        'renal_disease': 1,
        'heart_rate_min': 74,
        'heart_rate_max': 133,
        'heart_rate_mean': 103.99,
        'sbp_min': 61,
        'sbp_max': 193,
        'sbp_mean': 114.23,
        'dbp_min': 39,
        'dbp_max': 101,
        'dbp_mean': 60.24,
        'resp_rate_min': 7,
        'resp_rate_max': 25,
        'resp_rate_mean': 15.45,
        'spo2_min': 85,
        'spo2_max': 100,
        'spo2_mean': 97.90,
        'coma': 0,
        'albumin': 2.0,
        'race': "Black or African American",
        'antibiotic_vancomycin': "Yes",
        'gender': "Female"
    }
    
    example3 = {
        'max_age': 72,
        'los_icu': 5.45,
        'sofa_score': 10,
        'avg_urineoutput': 25.3,
        'glucose_min': 110,
        'glucose_max': 210,
        'glucose_average': 158,
        'sodium_max': 152,
        'sodium_min': 125,
        'sodium_average': 138.5,
        'diabetes_without_cc': 1,
        'diabetes_with_cc': 1,
        'severe_liver_disease': 1,
        'aids': 0,
        'renal_disease': 1,
        'heart_rate_min': 60,
        'heart_rate_max': 145,
        'heart_rate_mean': 95.5,
        'sbp_min': 70,
        'sbp_max': 160,
        'sbp_mean': 110.5,
        'dbp_min': 45,
        'dbp_max': 95,
        'dbp_mean': 65.7,
        'resp_rate_min': 12,
        'resp_rate_max': 38,
        'resp_rate_mean': 26.3,
        'spo2_min': 80,
        'spo2_max': 98,
        'spo2_mean': 92.4,
        'coma': 1,
        'albumin': 2.2,
        'race': "White",
        'antibiotic_vancomycin': "Yes",
        'gender': "Male"
    }
    
    return example1, example2, example3

# Main function to run the app
def main():
    # Add background
    add_bg_from_gradient()
    
    # Load animation
    load_animation()
    
    # Create header
    create_header()
    
    # Load model and scaler
    model, scaler = load_model()
    model_loaded = model is not None and scaler is not None
    
    # Initialize session state for form values
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {}
    
    # Create sidebar for navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=80)
        st.markdown("## Navigation")
        page = st.radio("", ["Prediction Tool", "Model Information", "About"])
        
        st.markdown("---")
        st.markdown("### Features Used")
        st.info("""
        - Patient Demographics
        - Vital Signs
        - Lab Values
        - Clinical Scores
        - Comorbidities
        """)
        
        st.markdown("---")
        st.markdown("### Need Help?")
        with st.expander("How to use this tool"):
            st.markdown("""
            1. Enter patient information in the form
            2. Click the 'Predict Mortality Risk' button
            3. Review the results and predictions
            4. Explore model insights for further understanding
            """)

    # Main content based on selected page
    if page == "Prediction Tool" and model_loaded:
        st.markdown('<div class="section-header">Patient Data Input</div>', unsafe_allow_html=True)
        
        # Example patient buttons
        st.markdown("### Quick Patient Examples")
        st.markdown("Click one of the buttons below to load example patient data:")
        
        # Get example data
        example1, example2, example3 = get_example_data()
        
        # Create a row of buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            example1_btn = st.button("Low Risk Patient", key="example1_btn", help="Load Low-risk patient example")
        
        with col2:
            example2_btn = st.button("High Risk Patient", key="example2_btn", help="Load High-risk patient example")
        
        with col3:
            example3_btn = st.button("Medium Risk Patient", key="example3_btn", help="Load Medium patient example")
        
        # Handle button clicks to set session state
        if example1_btn:
            st.session_state.form_values = example1
        
        if example2_btn:
            st.session_state.form_values = example2
        
        if example3_btn:
            st.session_state.form_values = example3
        
        # Create columns for input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Patient demographics
            st.subheader("Demographics")
            age = dynamic_input("Age", "max_age")
            gender = st.selectbox("Gender", ["Male", "Female"], 
                                 index=0 if get_value('gender', "Male") == "Male" else 1)
            race = st.selectbox("Race", ["White", "Black or African American", "Hispanic or Latin", "Others race"],
                              index=["White", "Black or African American", "Hispanic or Latin", "Others race"].index(get_value('race', "White")))
            
            # Clinical scores
            st.subheader("Clinical Scores")
            sofa_score = dynamic_input("SOFA Score", "sofa_score")
            coma = st.radio("Coma", ["No", "Yes"], 
                           index=1 if get_value('coma', 0) == 1 else 0)
            
        with col2:
            # Vital signs
            st.subheader("Vital Signs")
            heart_rate_min = dynamic_input("Minimum Heart Rate", "heart_rate_min")
            heart_rate_max = dynamic_input("Maximum Heart Rate", "heart_rate_max")
            heart_rate_mean = dynamic_input("Mean Heart Rate", "heart_rate_mean", step=0.1)
            
            sbp_min = dynamic_input("Minimum Systolic BP", "sbp_min")
            sbp_max = dynamic_input("Maximum Systolic BP", "sbp_max")
            sbp_mean = dynamic_input("Mean Systolic BP", "sbp_mean", step=0.1)
            
            dbp_min = dynamic_input("Minimum Diastolic BP", "dbp_min")
            dbp_max = dynamic_input("Maximum Diastolic BP", "dbp_max")
            dbp_mean = dynamic_input("Mean Diastolic BP", "dbp_mean", step=0.1)
            
        with col3:
            # Lab values and other parameters
            st.subheader("Laboratory Values")
            resp_rate_min = dynamic_input("Minimum Respiratory Rate", "resp_rate_min")
            resp_rate_max = dynamic_input("Maximum Respiratory Rate", "resp_rate_max")
            resp_rate_mean = dynamic_input("Mean Respiratory Rate", "resp_rate_mean", step=0.1)
            
            spo2_min = dynamic_input("Minimum SpO2", "spo2_min")
            spo2_max = dynamic_input("Maximum SpO2", "spo2_max")
            spo2_mean = dynamic_input("Mean SpO2", "spo2_mean", step=0.1)
            
            glucose_min = dynamic_input("Minimum Glucose", "glucose_min")
            glucose_max = dynamic_input("Maximum Glucose", "glucose_max")
            glucose_average = dynamic_input("Average Glucose", "glucose_average", step=0.1)
            
            sodium_min = dynamic_input("Minimum Sodium", "sodium_min")
            sodium_max = dynamic_input("Maximum Sodium", "sodium_max")
            sodium_average = dynamic_input("Average Sodium", "sodium_average", step=0.1)
            
            albumin = dynamic_input("Albumin", "albumin", step=0.1)
            
        # Additional row for more parameters
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Comorbidities")
            diabetes_without_cc = st.radio("Diabetes Without Complications", ["No", "Yes"], 
                                         index=1 if get_value('diabetes_without_cc', 0) == 1 else 0)
            diabetes_with_cc = st.radio("Diabetes With Complications", ["No", "Yes"], 
                                      index=1 if get_value('diabetes_with_cc', 0) == 1 else 0)
            
        with col2:
            severe_liver_disease = st.radio("Severe Liver Disease", ["No", "Yes"], 
                                          index=1 if get_value('severe_liver_disease', 0) == 1 else 0)
            renal_disease = st.radio("Renal Disease", ["No", "Yes"], 
                                   index=1 if get_value('renal_disease', 0) == 1 else 0)
            aids = st.radio("AIDS", ["No", "Yes"], 
                          index=1 if get_value('aids', 0) == 1 else 0)
            
        with col3:
            st.subheader("Other Variables")
            los_icu = dynamic_input("ICU Length of Stay (days)", "los_icu", step=0.1)
            avg_urineoutput = dynamic_input("Average Urine Output (ml/hr)", "avg_urineoutput", step=0.1)
            antibiotic_vancomycin = st.radio("Vancomycin Treatment", ["No", "Yes"], 
                                          index=1 if get_value('antibiotic_vancomycin', "No") == "Yes" else 0)
            
        # Create a prediction button with loading animation
        st.markdown("---")
        predict_col1, predict_col2 = st.columns([1, 1])
        
        with predict_col1:
            predict_button = st.button("Predict Mortality Risk", key="predict")
        
        # Make prediction when button is clicked
        if predict_button:
            # Create a loading animation
            with st.spinner("Computing prediction..."):
                time.sleep(1.5)  # Simulate computation time
                
                # Create input dataframe for prediction
                input_data = {
                    'max_age': age,
                    'los_icu': los_icu,
                    'sofa_score': sofa_score,
                    'avg_urineoutput': avg_urineoutput,
                    'glucose_min': glucose_min,
                    'glucose_max': glucose_max,
                    'glucose_average': glucose_average,
                    'sodium_max': sodium_max,
                    'sodium_min': sodium_min,
                    'sodium_average': sodium_average,
                    'diabetes_without_cc': 1 if diabetes_without_cc == "Yes" else 0,
                    'diabetes_with_cc': 1 if diabetes_with_cc == "Yes" else 0,
                    'severe_liver_disease': 1 if severe_liver_disease == "Yes" else 0,
                    'aids': 1 if aids == "Yes" else 0,
                    'renal_disease': 1 if renal_disease == "Yes" else 0,
                    'heart_rate_min': heart_rate_min,
                    'heart_rate_max': heart_rate_max,
                    'heart_rate_mean': heart_rate_mean,
                    'sbp_min': sbp_min,
                    'sbp_max': sbp_max,
                    'sbp_mean': sbp_mean,
                    'dbp_min': dbp_min,
                    'dbp_max': dbp_max,
                    'dbp_mean': dbp_mean,
                    'resp_rate_min': resp_rate_min,
                    'resp_rate_max': resp_rate_max,
                    'resp_rate_mean': resp_rate_mean,
                    'spo2_min': spo2_min,
                    'spo2_max': spo2_max,
                    'spo2_mean': spo2_mean,
                    'coma': 1 if coma == "Yes" else 0,
                    'albumin': albumin,
                    'race_Black or African American': 1 if race == "Black or African American" else 0,
                    'race_Hispanic or Latin': 1 if race == "Hispanic or Latin" else 0,
                    'race_Others race': 1 if race == "Others race" else 0,
                    'race_White': 1 if race == "White" else 0,
                    'antibiotic_Vancomycin': 1 if antibiotic_vancomycin == "Yes" else 0,
                    'gender_F': 1 if gender == "Female" else 0,
                    'gender_M': 1 if gender == "Male" else 0,
                }
                
                # Convert to dataframe
                input_df = pd.DataFrame([input_data])
                
                # Scale the input data
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]

                prediction_prob = model.predict_proba(input_scaled)[0][1]  # Probability of class 1
            
            # Display prediction results
            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

            result_col1, result_col2 = st.columns([2, 1])

            with result_col1:
                # Display gauge chart
                gauge_chart = create_gauge_chart(prediction_prob)
                st.plotly_chart(gauge_chart, use_container_width=True)

                # Final prediction
                risk_text = "High Mortality Risk" if prediction == 1 else "Low Mortality Risk"
                color_class = "high-risk" if prediction == 1 else "low-risk"
                st.markdown(
                    f'<div class="prediction-result {color_class}">{risk_text}<br>Probability: {prediction_prob:.2%}</div>',
                    unsafe_allow_html=True
                )

            with result_col2:
                st.subheader("Key Risk Indicators")
                risk_factors = []

                # Smart threshold logic
                thresholds = {
                    "sofa_score": (5, "High SOFA score"),
                    "coma": ("Yes", "Coma state"),
                    "resp_rate_mean": (22, "Elevated respiratory rate"),
                    "heart_rate_max": (110, "Tachycardia"),
                    "sbp_min": (90, "Hypotension"),
                    "glucose_max": (180, "Hyperglycemia"),
                    "albumin": (3.0, "Low albumin"),
                    "avg_urineoutput": (50, "Low urine output"),
                    "max_age": (75, "Advanced age")
                }

                # Numeric checks
                if sofa_score > thresholds["sofa_score"][0]:
                    risk_factors.append(f"{thresholds['sofa_score'][1]} ({sofa_score})")
                if coma == thresholds["coma"][0]:
                    risk_factors.append(thresholds["coma"][1])
                if resp_rate_mean > thresholds["resp_rate_mean"][0]:
                    risk_factors.append(f"{thresholds['resp_rate_mean'][1]} ({resp_rate_mean})")
                if heart_rate_max > thresholds["heart_rate_max"][0]:
                    risk_factors.append(f"{thresholds['heart_rate_max'][1]} ({heart_rate_max})")
                if sbp_min < thresholds["sbp_min"][0]:
                    risk_factors.append(f"{thresholds['sbp_min'][1]} (SBP {sbp_min})")
                if glucose_max > thresholds["glucose_max"][0]:
                    risk_factors.append(f"{thresholds['glucose_max'][1]} ({glucose_max})")
                if albumin < thresholds["albumin"][0]:
                    risk_factors.append(f"{thresholds['albumin'][1]} ({albumin})")
                if avg_urineoutput < thresholds["avg_urineoutput"][0]:
                    risk_factors.append(f"{thresholds['avg_urineoutput'][1]} ({avg_urineoutput})")
                if age > thresholds["max_age"][0]:
                    risk_factors.append(f"{thresholds['max_age'][1]} ({age})")

                # Comorbidity checks
                if renal_disease == "Yes":
                    risk_factors.append("Renal disease")
                if severe_liver_disease == "Yes":
                    risk_factors.append("Severe liver disease")

                # Display results
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("No major risk factors identified")


            
            # Show detailed analysis
            st.markdown("---")
            st.markdown('<div class="section-header">Model Insights</div>', unsafe_allow_html=True)
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                # Feature importance plot
                st.subheader("Feature Importance")
                fig = plot_feature_importance(model, input_df.columns)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance visualization not available for this model.")
            
            with insight_col2:
                # Clinical recommendations
                st.subheader("Clinical Recommendations")
                
                if prediction == 1:
                    st.error("""
                    ### High Mortality Risk Detected
                    
                    Recommended actions:
                    - Consider ICU escalation if not already there
                    - Immediate clinical intervention
                    - Consider palliative care consult if appropriate
                    - Discuss goals of care with patient/family
                    - Optimize organ support therapies
                    """)
                else:
                    st.success("""
                    ### Low Mortality Risk Detected
                    
                    Recommended actions:
                    - Continue standard monitoring
                    - Address any specific clinical issues
                    - Consider discharge planning if clinically appropriate
                    - Monitor for any changes in clinical status
                    """)
                
                st.info("These recommendations are based on general ICU management protocols. Clinical judgment should always take precedence.")
            
    elif page == "Model Information":
        st.markdown('<div class="section-header">About the Model</div>', unsafe_allow_html=True)
        
        st.write("""
        This application uses a Stacking Classifier model to predict ICU mortality risk. 
        The model combines predictions from multiple base models using a meta-learner for improved performance.
        """)
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.subheader("Model Architecture")
            st.write("""
            The Stacking Classifier ensemble includes:
            - Random Forest Classifier (200 trees, max depth 20)
            - XGBoost Classifier (200 estimators)
            - Support Vector Machine (C=10)
            - Meta-learner: Logistic Regression
            """)
            
            st.subheader("Training Dataset")
            st.write("""
            - Dataset Size: Clinical data from ICU patient records
            - Features: 39 clinical variables including vital signs, lab values, and comorbidities
            - Target: Hospital mortality
            - SMOTE applied to handle class imbalance
            """)
            
        with model_col2:
            st.subheader("Model Performance")
            
            # Create performance metrics based on the training file
            metrics = {
                'Accuracy': 0.89,
                'ROC-AUC': 0.95,
                'Precision': 0.90,
                'Recall': 0.87,
                'F1 Score': 0.88
            }
            
            # Create metrics visualization
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color='royalblue',
                text=[f"{v:.2f}" for v in metrics.values()],
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Metric",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "About":
        st.markdown('<div class="section-header">About this Application</div>', unsafe_allow_html=True)
        
        st.write("""
        This application was designed to help clinicians assess mortality risk in ICU patients using advanced machine learning techniques.
        The tool analyzes patient data to provide risk predictions and clinical insights for decision support.
        
        ### ICU Mortality Risk Assessment
        
        Predicting mortality risk in ICU patients is critical for resource allocation, treatment planning, and family discussions.
        Early identification of high-risk patients can help guide interventions and improve outcomes.
        
        ### How to Use This Tool
        
        1. Enter patient clinical data in the "Prediction Tool" section
        2. Click "Predict Mortality Risk" to get a risk assessment
        3. Review the results and recommendations
        4. Explore model insights for better understanding
        
        ### Important Note
        
        This tool is designed to assist clinical decision-making but should not replace clinical judgment.
        Always consult with healthcare professionals for diagnosis and treatment decisions.
        """)
        
        st.markdown('<div class="section-header">Dataset Information</div>', unsafe_allow_html=True)
        
        st.write("""
        The model was trained on data from ICU patients with the following characteristics:
        
        - Features include vital signs, laboratory values, comorbidities, and demographic information
        - Data was preprocessed with standard scaling and missing value imputation
        - Class imbalance was addressed using SMOTE technique
        """)
        
        # Show correlation heatmap
        st.subheader("Top Feature Correlations with Mortality")
        
        # Sample correlation data for visualization based on common clinical knowledge
        corr_data = { 
            'Feature': ['coma', 'sofa_score', 'resp_rate_mean', 'resp_rate_max', 'heart_rate_max', 
                'severe_liver_disease', 'antibiotic_Vancomycin', 'heart_rate_mean', 'los_icu', 
                'max_age', 'glucose_min', 'spo2_mean', 'sodium_min', 'dbp_mean', 'avg_urineoutput', 
                'albumin', 'spo2_min', 'sbp_mean', 'dbp_min', 'sbp_min'], 
            'Correlation': [0.149961, 0.124907, 0.123988, 0.111966, 0.102559, 
                    0.092304, 0.090209, 0.073573, 0.072592, 0.058536, 
                    -0.051236, -0.074220, -0.074336, -0.105440, -0.113512, 
                    -0.125716, -0.127147, -0.132240, -0.161527, -0.177369]
}
        
        corr_df = pd.DataFrame(corr_data)
        
        # Create horizontal bar chart
        fig = px.bar(
            corr_df,
            x='Correlation',
            y='Feature',
            orientation='h',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            title='Top Feature Correlations with Mortality Outcome'
        )
        
        fig.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">© 2025 ICU Mortality Predictor | Developed using Streamlit | Not for clinical use without validation</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()