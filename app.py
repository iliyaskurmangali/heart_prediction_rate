import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
best_model = joblib.load('best_model.joblib')

# Define the Streamlit app
def main():
    st.title("Heart Disease Prediction")

    st.write("""
    Enter the following information to predict whether you have heart disease or not:
    """)

    # Input fields for each feature
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain_type = st.number_input("Chest pain type", min_value=1, max_value=4, value=1)
    bp = st.number_input("Resting Blood Pressure (BP)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
    fbs_over_120 = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ekg_results = st.number_input("Resting Electrocardiographic Results", min_value=0, max_value=2, value=0)
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope_of_st = st.number_input("Slope of the Peak Exercise ST Segment", min_value=0, max_value=2, value=0)
    number_of_vessels_fluro = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thallium = st.number_input("Thallium Stress Test Result", min_value=1, max_value=7, value=3)

    # Button to trigger prediction
    if st.button("Predict"):
        # Create a dictionary with input data
        input_data = {
            'Age': [age],
            'Sex': [sex],
            'Chest pain type': [chest_pain_type],
            'BP': [bp],
            'Cholesterol': [cholesterol],
            'FBS over 120': [fbs_over_120],
            'EKG results': [ekg_results],
            'Max HR': [max_hr],
            'Exercise angina': [exercise_angina],
            'ST depression': [st_depression],
            'Slope of ST': [slope_of_st],
            'Number of vessels fluro': [number_of_vessels_fluro],
            'Thallium': [thallium]
        }

        try:
            # Preprocess the input data manually
            X_processed = pd.DataFrame(input_data)

            # Make prediction
            prediction = best_model.predict(X_processed)[0]

            # Decode the prediction
            prediction_decoded = "No Heart Disease" if prediction == 0 else "Heart Disease"

            # Display the prediction
            st.write(f"Prediction: **{prediction_decoded}**")
        except ValueError as e:
            st.error(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()
