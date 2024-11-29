import streamlit as st
import joblib
import pandas as pd

# Load the trained model (replace with your model's path)
model = joblib.load('hydrogen_leakage_model.pkl')

# Title of the app
st.title("Hydrogen Leakage Detection System")

# Description
st.write("This is a hydrogen leakage detection system powered by machine learning. Enter the required data below to get predictions.")

# User input for Hydrogen Level and Rate of Change
hydrogen_level = st.number_input("Enter Hydrogen Level (ppm)", min_value=0.0, max_value=100.0, step=0.1)
rate_of_change = st.number_input("Enter Rate of Change (ppm/s)", min_value=-100.0, max_value=100.0, step=0.1)

# Button to make predictions
if st.button("Predict"):
    if hydrogen_level is not None and rate_of_change is not None:
        # Convert the features into the same format as the model expects (DataFrame)
        input_data = pd.DataFrame([[hydrogen_level, rate_of_change]], columns=['Hydrogen Level (ppm)', 'Rate of Change (ppm/s)'])
        
        # Make prediction using the model
        prediction = model.predict(input_data)

        # Map the prediction to readable results
        leakage_status = 'Leakage' if prediction[0] == 1 else 'No Leakage'

        # Display the result
        st.write(f"Prediction: **{leakage_status}**")
    else:
        st.error("Please enter valid values for both features.")



# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model (replace with your model's path)
# model = joblib.load('hydrogen_leakage_model.pkl')

# # If you used scaling, load the scaler too (optional)
# # scaler = joblib.load('scaler.pkl')

# # Define a route for the home page (GET request)
# @app.route('/', methods=['GET'])
# def home():
#     return "Welcome to the Hydrogen Leakage Detection API! Use the /predict endpoint to make predictions."

# # Define a route for making predictions (POST request)
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the data from the POST request
#         data = request.get_json()

#         # Extract features from the request
#         hydrogen_level = data['Hydrogen Level (ppm)']
#         rate_of_change = data['Rate of Change (ppm/s)']

#         # Convert the features into the same format as the model expects (DataFrame)
#         input_data = pd.DataFrame([[hydrogen_level, rate_of_change]], columns=['Hydrogen Level (ppm)', 'Rate of Change (ppm/s)'])

#         # If you applied scaling during model training, apply the scaler here
#         # input_data_scaled = scaler.transform(input_data)

#         # Make prediction using the model
#         prediction = model.predict(input_data)

#         # Map the prediction to readable results
#         leakage_status = 'Leakage' if prediction[0] == 1 else 'No Leakage'

#         # Return the prediction as a JSON response
#         return jsonify({'leakage_status': leakage_status})

#     except Exception as e:
#         # Handle errors gracefully
#         return jsonify({'error': str(e)}), 400

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
