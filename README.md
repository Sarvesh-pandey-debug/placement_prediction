# placement_prediction 
# code for deploye model using streamlit
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('placement_model.pkl')

# Title and instructions
st.title("Placement Prediction App")
st.write("Enter the following details to predict if the student will be placed.")

# Input fields for IQ and Percentage
iq = st.number_input("Enter IQ Level", min_value=80, max_value=160, value=100, step=1)
percentage = st.number_input("Enter Percentage", min_value=0.0, max_value=100.0, value=75.0)

# Predict button
if st.button("Predict Placement"):
    # Prepare the features as an array for prediction
    features = np.array([[iq, percentage]])

    # Get the prediction (1 for placed, 0 for not placed)
    prediction = model.predict(features)[0]

    # Display the result
    if prediction == 1:
        st.success("The student is likely to be placed.")
    else:
        st.error("The student is not likely to be placed.")
