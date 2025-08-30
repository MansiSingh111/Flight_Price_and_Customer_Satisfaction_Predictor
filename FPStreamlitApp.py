import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Flight Price Predictor", layout="centered")
st.title("✈️ Flight Price Predictor")

@st.cache_resource
def load_xgb_model():
    with open('models.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_xgb_model()

# --- Feature List ---
feature_names = [
    'Duration_in_minutes', 'Number_of_Stops', 'Journey_year', 'Journey_month', 'Journey_day',
    'Dep_hour', 'Dep_minute', 'Arr_hour', 'Arr_minute',
    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
    'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi',
    'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',
    'Airline_Multiple carriers', 'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet', 'Airline_Trujet',
    'Airline_Vistara', 'Airline_Vistara Premium economy'
]

st.header("Enter Flight Details:")

# --- Numeric Inputs ---
Duration_in_minutes = st.number_input("Duration (minutes)", min_value=0, max_value=2000, value=120)
Number_of_Stops = st.selectbox("Number of Stops", [0, 1, 2, 3, 4])
Journey_year = st.selectbox("Year", [2019])
Journey_month = st.selectbox("Month", list(range(1, 13)), index=0)
Journey_day = st.slider("Day of Month", min_value=1, max_value=31, value=1)
Dep_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=12)
Dep_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0)
Arr_hour = st.slider("Arrival Hour", min_value=0, max_value=23, value=16)
Arr_minute = st.slider("Arrival Minute", min_value=0, max_value=59, value=0)

# --- One-hot Selection for Source ---
Source_city = st.selectbox("Source City", ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'])

# --- One-hot Selection for Destination ---
Destination = st.selectbox(
    "Destination City", ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
)


# --- One-hot Selection for Airline ---
Airline = st.selectbox(
    "Airline",
    ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
     'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy']
)

# --- Final Input Array in Model Order ---
input_features = [
    Airline, Source_city, Destination, Duration_in_minutes,
       Number_of_Stops, Journey_year, Journey_month, Journey_day,
       Dep_hour, Dep_minute, Arr_hour, Arr_minute
] 

input_df = pd.DataFrame([input_features], columns = ['Airline', 'Source', 'Destination', 'Duration_in_minutes',
       'Number_of_Stops', 'Journey_year', 'Journey_month', 'Journey_day',
       'Dep_hour', 'Dep_minute', 'Arr_hour', 'Arr_minute'])


# --- Predict Button ---
if st.button("Predict Price"):
    pred_price = model.predict(input_df)
    st.success(f"Predicted Flight Price: ₹ {pred_price[0]:,.2f}")
