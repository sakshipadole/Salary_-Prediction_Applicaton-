import streamlit as st
import pickle
import numpy as np

# Set page title
st.set_page_config(page_title="Salary Predictor", layout="centered")

# 1. Load the trained model
@st.cache_resource
def load_model():
    with open('model (3).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# 2. App Interface
st.title("💰 Salary Prediction App")
st.write("This app predicts salary based on **Years of Experience** using a Linear Regression model.")

# Create an input for Years of Experience
years_exp = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# 3. Prediction Logic
if st.button("Predict Salary"):
    # Reshape input for scikit-learn (needs to be 2D)
    input_data = np.array([[years_exp]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"The estimated salary is: **${prediction[0]:,.2f}**")

# Footer
st.divider()
st.info("Model details: scikit-learn LinearRegression | Feature: YearsExperience")
