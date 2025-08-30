import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Customer Satisfaction Predictor", layout="centered")
st.title("✈️ Customer Satisfaction Predictor")

@st.cache_resource
def load_rf_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_rf_model()

st.header("Enter Passenger and Flight Details:")

# --- Numeric Inputs ---
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

# --- Encoded/Categorical Inputs ---
Class_encoded = st.selectbox("Travel Class", [0, 1, 2], format_func=lambda x: ["Eco", "Business", "Eco Plus"][x])
Type_of_Travel_encoded = st.selectbox("Type of Travel", [0, 1], format_func=lambda x: ["Business", "Personal"][x])
Gender_Male = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])
Customer_Type_disloyal_Customer = st.selectbox("Customer Type", [0, 1], format_func=lambda x: ["Loyal", "Disloyal"][x])

# --- Final Input Array in Model Order ---
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
