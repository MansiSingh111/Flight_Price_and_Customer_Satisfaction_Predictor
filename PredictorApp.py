import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Flight & Customer Prediction", layout="centered")

# --- Sidebar navigation ---
page = st.sidebar.selectbox("Select Predictor", ["Flight Price Predictor", "Customer Satisfaction Predictor"])

# --- Flight Price Predictor Page ---
if page == "Flight Price Predictor":
    st.title("✈️ Flight Price Predictor")

    @st.cache_resource
    def load_xgb_model():
        with open('models.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    model = load_xgb_model()

    st.header("Enter Flight Details:")

    Duration_in_minutes = st.number_input("Duration (minutes)", min_value=0, max_value=2000, value=120)
    Number_of_Stops = st.selectbox("Number of Stops", [0, 1, 2, 3, 4])
    Journey_year = st.selectbox("Year", [2019])
    Journey_month = st.selectbox("Month", list(range(1, 13)), index=0)
    Journey_day = st.slider("Day of Month", min_value=1, max_value=31, value=1)
    Dep_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=12)
    Dep_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0)
    Arr_hour = st.slider("Arrival Hour", min_value=0, max_value=23, value=16)
    Arr_minute = st.slider("Arrival Minute", min_value=0, max_value=59, value=0)
    Source_city = st.selectbox("Source City", ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
    Destination = st.selectbox("Destination City", ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi'])
    Airline = st.selectbox(
        "Airline",
        ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
         'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy']
    )

    input_features = [
        Airline, Source_city, Destination, Duration_in_minutes,
        Number_of_Stops, Journey_year, Journey_month, Journey_day,
        Dep_hour, Dep_minute, Arr_hour, Arr_minute
    ] 

    input_df = pd.DataFrame([input_features], columns=[
        'Airline', 'Source', 'Destination', 'Duration_in_minutes',
        'Number_of_Stops', 'Journey_year', 'Journey_month', 'Journey_day',
        'Dep_hour', 'Dep_minute', 'Arr_hour', 'Arr_minute'
    ])

    if st.button("Predict Price"):
        pred_price = model.predict(input_df)
        st.success(f"Predicted Flight Price: ₹ {pred_price[0]:,.2f}")

# --- Customer Satisfaction Predictor Page ---
else:
    st.title("✈️ Customer Satisfaction Predictor")

    @st.cache_resource
    def load_rf_model():
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    model = load_rf_model()

    st.header("Enter Passenger and Flight Details:")

    Age = st.number_input("Age", min_value=0, max_value=100, value=30)
    Flight_Distance = st.number_input("Flight Distance", min_value=0, max_value=10000, value=500)
    Inflight_wifi_service = st.slider("Inflight WiFi Service (0-5)", min_value=0, max_value=5, value=3)
    Departure_Arrival_time_convenient = st.slider("Departure/Arrival Time Convenient (0-5)", min_value=0, max_value=5, value=3)
    Ease_of_Online_booking = st.slider("Ease of Online Booking (0-5)", min_value=0, max_value=5, value=3)
    Gate_location = st.slider("Gate Location (0-5)", min_value=0, max_value=5, value=3)
    Food_and_drink = st.slider("Food and Drink (0-5)", min_value=0, max_value=5, value=3)
    Online_boarding = st.slider("Online Boarding (0-5)", min_value=0, max_value=5, value=3)
    Seat_comfort = st.slider("Seat Comfort (0-5)", min_value=0, max_value=5, value=3)
    Inflight_entertainment = st.slider("Inflight Entertainment (0-5)", min_value=0, max_value=5, value=3)
    On_board_service = st.slider("On-board Service (0-5)", min_value=0, max_value=5, value=3)
    Leg_room_service = st.slider("Leg Room Service (0-5)", min_value=0, max_value=5, value=3)
    Baggage_handling = st.slider("Baggage Handling (0-5)", min_value=0, max_value=5, value=3)
    Checkin_service = st.slider("Check-in Service (0-5)", min_value=0, max_value=5, value=3)
    Inflight_service = st.slider("Inflight Service (0-5)", min_value=0, max_value=5, value=3)
    Cleanliness = st.slider("Cleanliness (0-5)", min_value=0, max_value=5, value=3)
    Departure_Delay_in_Minutes = st.number_input("Departure Delay (minutes)", min_value=0, max_value=600, value=0)
    Arrival_Delay_in_Minutes = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=600, value=0)
    Class_encoded = st.selectbox("Travel Class", [0, 1, 2], format_func=lambda x: ["Eco", "Business", "Eco Plus"][x])
    Type_of_Travel_encoded = st.selectbox("Type of Travel", [0, 1], format_func=lambda x: ["Business", "Personal"][x])
    Gender_Male = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])
    Customer_Type_disloyal_Customer = st.selectbox("Customer Type", [0, 1], format_func=lambda x: ["Loyal", "Disloyal"][x])

    input_features = [
        Age, Flight_Distance, Inflight_wifi_service, Departure_Arrival_time_convenient,
        Ease_of_Online_booking, Gate_location, Food_and_drink, Online_boarding,
        Seat_comfort, Inflight_entertainment, On_board_service, Leg_room_service,
        Baggage_handling, Checkin_service, Inflight_service, Cleanliness,
        Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes,
        Class_encoded, Type_of_Travel_encoded,
        Gender_Male, Customer_Type_disloyal_Customer
    ]

    input_array = np.array(input_features).reshape(1, -1)

    if st.button("Predict Satisfaction"):
        pred_satisfaction = model.predict(input_array)
        label = "Satisfied" if pred_satisfaction[0] == 1 else "Not Satisfied"
        st.success(f"Predicted Customer Satisfaction: {label}")
