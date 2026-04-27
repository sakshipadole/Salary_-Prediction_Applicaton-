import streamlit as st
import pickle
import numpy as np
import sklearn

# Set page title and favicon
st.set_page_config(page_title="Salary Predictor", page_icon="💰")

# 1. Properly Cache the Model to prevent loading loops
@st.cache_resource
def load_trained_model():
    try:
        # Note: Ensure the filename matches exactly what you uploaded
        with open('model (3).pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# 2. User Interface
st.title("Salary Prediction App")
st.info(f"Running on scikit-learn version: {sklearn.__version__}")

if model is not None:
    years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0)

    if st.button("Predict Salary"):
        # Reshape for single feature prediction
        features = np.array([[years_exp]])
        prediction = model.predict(features)
        
        st.success(f"Estimated Salary: ${prediction[0]:,.2f}")
else:
    st.warning("Model file not found. Please ensure 'model (3).pkl' is in the same folder as app.py")
