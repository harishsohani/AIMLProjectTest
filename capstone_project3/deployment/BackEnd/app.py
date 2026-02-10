
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app with a name
pred_mainteanance_api = Flask ("Engine Maintenance Predictor")

# Load the trained churn prediction model
model = joblib.load ("???.joblib")

# Define a route for the home page
@pred_mainteanance_api.get ('/')
def home ():
    return "Welcome to the Engine Maintenance Prediction!"

# Define an endpoint to predict sales for Super Kart
@pred_mainteanance_api.post ('/v1/EngPredMaintenance')
def predict_need_maintenance ():
    # Get JSON data from the request
    engine_sensor_inputs = request.get_json ()

    import datetime

    current_year = datetime.datetime.now ().year   # dynamic current year

    # Extract relevant features from the input data
    data_info = {
        'Engine_rpm'                : engine_sensor_inputs ['Engine_rpm'],
        'Lub_oil_pressure'          : engine_sensor_inputs ['Lub_oil_pressure'],
        'Fuel_pressure'             : engine_sensor_inputs ['Fuel_pressure'],
        'Coolant_pressure'          : engine_sensor_inputs ['Coolant_pressure'],
        'lub_oil_temp'              : engine_sensor_inputs ['lub_oil_temp'],
        'Coolant_temp'              : engine_sensor_inputs ['Coolant_temp']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame ([data_info])

    # Enforce types - convert all to float
    input_data = input_data.astype (float)

    # Make prediction using the trained model
    predicted_sales = model.predict (input_data).tolist ()[0]

    # Return the prediction as a JSON response
    return jsonify ({'NeedsMaintenance': predicted_sales})


# Run the Flask app
if __name__ == "__main__":
    import os
    port = int (os.environ.get("PORT", 7860))
    pred_mainteanance_api.run(host="0.0.0.0", port=port)
