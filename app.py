import streamlit as st
import pickle
import numpy as np

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Machine Learning Model App")

st.write("Enter feature values:")

# 🔍 Try dynamic feature handling (fallback: manual)
try:
    n_features = model.n_features_in_
except:
    n_features = 3  # default (change if needed)

inputs = []

# Create dynamic input fields
for i in range(n_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

input_array = np.array([inputs])

if st.button("Predict"):
    try:
        prediction = model.predict(input_array)
        st.success(f"Prediction: {prediction[0]}")
        
        # If classification model
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_array)
            st.write("Prediction Probability:", proba)
            
    except Exception as e:
        st.error(f"Error: {e}")
