
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
    page_title="Predictive Maintenenace App"  #,
    #layout="wide"
)


# Download and load the model
model_path = hf_hub_download(
    repo_id="harishsohani/AIMLProjectTest",
    #### Final name will be ####
    # hf_repo_id = "harishsohani/AIMLPredictMaintenance"
    filename="best_eng_fail_pred_model.joblib"
    )
model = joblib.load(model_path)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🏖️ Predict Maintenance")
st.write("The Predict Maintenance app is a tool to predict if an Engine needs any maintenance based on provided operating sensor parameters.")
st.write("Fill in the details below and click **Predict** to see if the Engine needs maintenance to prevent from failure.")



# ====================================
# Section : Capture Engine Parameters
# ====================================
st.subheader ("Engine Parameters")

rpm               = st.number_input(
    "Engine RPM (50.0 to 2500.0)",
    min_value=50,
    max_value=2500,
    value=735,
    step=10
)

oil_pressure      = st.number_input(
    "Lubricating oil pressure in kilopascals (kPa) (0.001 to 10.0)",
    min_value=0.001,
    max_value=10.0,
    value=3.300,
    step=0.001,
    format="%.3f"
)

fuel_pressure     = st.number_input(
    "Fuel Pressure in kilopascals (kPa) (0.01 to 25.0)",
    min_value=0.01,
    max_value=25.0,
    value=6.50,
    step=0.01,
    format="%.2f"
)

coolant_pressure  = st.number_input(
    "Coolant Pressure in kilopascals (kPa) (0.01 to 10.0)",
    min_value=0.01,
    max_value=10.0,
    value=2.25,
    step=0.10,
    format="%.2f"
)

lub_oil_temp      = st.number_input(
    "Lubricating oil Temperature in degrees Celsius (°C) (50.0 to 100.0)",
    min_value=50.0,
    max_value=100.0,
    value=75.0,
    step=0.1
)

coolant_temp      = st.number_input(
    "Coolant Temperature in degrees Celsius (°C)  (50.0 to 200.0)",
    min_value=50.0,
    max_value=200.0,
    value=75.0,
    step=1.0
)


# ==========================
# Single Value Prediction
# ==========================
if st.button("Check fo Maintenance"):

    # extract the data collected into a structure
    input_data = {
        'Engine_rpm'              : float(rpm),
        'Lub_oil_pressure'        : float(oil_pressure),
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


