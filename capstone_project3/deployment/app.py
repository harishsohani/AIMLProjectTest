
# import streamlit library for IO
import streamlit as st

# import pandas
import pandas as pd

# library to download fine from Hugging Face
from huggingface_hub import hf_hub_download

# library to load model
import joblib



# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenenace App",
    layout="wide"
)


# Download and load the model
model_path = hf_hub_download(
    repo_id="harishsohani/AIMLProjectTest",
    filename="best_eng_fail_pred_model.joblib"
    )
model = joblib.load(model_path)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🏖️ Predict for Maintenance")
st.write("Fill in the details below and click **Predict** to see if the Engine needs maintenance to prevent for failure.")



# ====================================
# Section : Capture Engine Parameters
# ====================================
st.subheader ("Engine Parameters")

rpm               = st.number_input ("Engine RPM (50.0 to 2500.0)", min_value=50, max_value=2500, value=735, step=10)
lub_oil_pressure  = st.number_input ("Lubricating oil pressure in kilopascals (kPa) (0.001 to 10.0)", min_value=0.001, max_value=10.0, value=3.30, step=0.001)
fuel_pressure     = st.number_input ("Fuel Pressure in kilopascals (kPa) (0.01 to 25.0)", min_value=0.01, max_value=25.0, value=6.5, step=0.01)
coolant_pressure  = st.number_input ("Coolant Pressure in kilopascals (kPa) (0.01 to 10.0)", min_value=0.01, max_value=10.0, value=2.25, step=0.1)
lub_oil_temp      = st.number_input ("Lubricating oil Temperature in degrees Celsius (°C) (50.0 to 100.0)", min_value=50.0, max_value=100.0, value=75.0, step=0.1)
coolant_temp      = st.number_input ("Coolant Temperature in degrees Celsius (°C)  (50.0 to 200.0)", min_value=50.0, max_value=200.0, value=75.0, step=1.0)


# ==========================
# Single Value Prediction
# ==========================
if st.button("Check fo Maintenance"):

    # extract the data collected into a structure
    input_data = {
        'Engine rpm'              : float(rpm),
        'Lub_oil_pressure'        : float(lub_oil_pressure),
        'Fuel_pressure'           : float(fuel_pressure),
        'Coolant_pressure'        : float(coolant_pressure),
        'lub_oil_temp'            : float(lub_oil_temp),
        'Coolant_temp'            : float(lub_oil_temp),
    }

    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]

    result = "Engine is **likely** needs maintenance." if prediction == 1 \
                 else "Engine does not need any maintenance"
    
    st.success(result)

    # Show the etails of data frame prepared from user input
    st.subheader("📦 Input Data Summary")
    st.json(input_df)


