import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

st.set_page_config(page_title="Flight Price & Passenger Satisfaction Prediction", layout="wide")

# --- LOAD DATA ---
@st.cache_resource
def load_flight_data():
    return pd.read_csv('Cleaned_Flight_Price data.csv')

@st.cache_resource
def load_customer_data():
    # Change if your customer satisfaction file is different
    return pd.read_csv('Cleaned_Passenger_Satisfaction_data.csv')

@st.cache_resource
def load_xgb_model():
    with open('models.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_rf_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# --- SIDEBAR ---
pages = ["Introduction",
         "Flight Price Predictor",
         "Flight Price EDA",
         "Passenger Satisfaction Predictor",
         "Passenger Satisfaction EDA"]
page = st.sidebar.selectbox("Navigate", pages)

# --- PAGE 1: INTRODUCTION ---
if page == "Introduction":
    st.title("✈️ Flight Price & Passenger Satisfaction Dashboard")
    st.markdown("""
        Welcome to the Flight Analytics and Prediction App!
        
        - **Flight Price Predictor**: Forecast your flight fare using flight details.
        - **Flight Price EDA**: Explore trends and patterns in flight pricing.
        - **Customer Satisfaction Predictor**: Predict airline passenger satisfaction.
        - **Customer Satisfaction EDA**: Investigate patterns behind customer satisfaction.

        Use the navigation sidebar to move between pages.
    """)

# --- PAGE 2: FLIGHT PRICE PREDICTOR ---
elif page == "Flight Price Predictor":
    st.title("✈️ Flight Price Predictor")
    df = load_flight_data()
    model = load_xgb_model()

    st.header("Enter Flight Details:")
    Duration_in_minutes = st.number_input("Duration (minutes)", min_value=0, max_value=2000, value=120)
    Number_of_Stops = st.selectbox("Number of Stops", sorted(df['Number_of_Stops'].unique().tolist()))
    Journey_year = st.selectbox("Year", sorted(df['Journey_year'].unique().tolist()))
    Journey_month = st.selectbox("Month", sorted(df['Journey_month'].unique().tolist()))
    Journey_day = st.slider("Day of Month", min_value=1, max_value=31, value=1)
    Dep_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=12)
    Dep_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0)
    Arr_hour = st.slider("Arrival Hour", min_value=0, max_value=23, value=16)
    Arr_minute = st.slider("Arrival Minute", min_value=0, max_value=59, value=0)
    Source_city = st.selectbox("Source City", sorted(df['Source'].unique().tolist()))
    Destination = st.selectbox("Destination City", sorted(df['Destination'].unique().tolist()))
    Airline = st.selectbox("Airline", sorted(df['Airline'].unique().tolist()))

    input_features = [
        Airline, Source_city, Destination, Duration_in_minutes, Number_of_Stops,
        Journey_year, Journey_month, Journey_day,
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

# --- PAGE 3: FLIGHT PRICE EDA ---
elif page == "Flight Price EDA":
    st.title("📊 Flight Price Exploratory Data Analysis")
    df = load_flight_data()
    sns.set_style("whitegrid")

    # 1. Bar: Average Price per Airline
    st.subheader("1. Average Price per Airline (Bar)")
    avg_prices = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10,4))
    avg_prices.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_ylabel('Average Price')
    st.pyplot(fig1)
    st.markdown("**Insight:** Jet Airways Business has a significantly higher average flight price than all other airlines, standing out as the most expensive option.")

    # 2. Barh: Top 10 Most Frequent Routes
    st.subheader("2. Top 10 Most Frequent Routes (Horizontal Bar)")
    route_counts = df.groupby(['Source', 'Destination']).size().sort_values(ascending=False).reset_index(name='Count')
    top_routes = route_counts.head(10)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.barh(top_routes['Source'] + " → " + top_routes['Destination'], top_routes['Count'], color='tan')
    ax2.set_xlabel('Number of Flights')
    ax2.invert_yaxis()
    st.pyplot(fig2)
    st.markdown("**Insight:** The route from Delhi to Cochin is by far the most frequently flown route, followed by kolkata to Bangalore.")

    # 3. Line: Price Trend by Month
    st.subheader("3. Average Price by Month (Line)")
    if 'Journey_month' in df.columns:
        price_by_month = df.groupby('Journey_month')['Price'].mean().reset_index()
        fig3, ax3 = plt.subplots(figsize=(8,4))
        ax3.plot(price_by_month['Journey_month'], price_by_month['Price'], marker='o', linestyle='-', color='green')
        ax3.set_xticks(price_by_month['Journey_month'])
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Price')
        st.pyplot(fig3)
        st.markdown("**Insight:** Flight prices peaked in march, dropped significantly in april, and then increased again in May and June.")

    # 4. Histogram: Distribution of Prices
    st.subheader("4. Distribution of Flight Prices (Histogram)")
    fig4, ax4 = plt.subplots(figsize=(8,4))
    ax4.hist(df['Price'], bins=40, color='cornflowerblue', edgecolor='black')
    ax4.set_xlabel('Price')
    ax4.set_ylabel('Frequency')
    st.pyplot(fig4)
    st.markdown("**Insight:** Most flight prices are concentrated below Rs.20,000 with a few exxpensive outliers.")

    # 5. Bar: Average Price by Number of Stops
    st.subheader("5. Average Price by Number of Stops (Bar)")
    avg_by_stops = df.groupby('Number_of_Stops')['Price'].mean()
    fig5, ax5 = plt.subplots(figsize=(6,4))
    avg_by_stops.plot(kind='bar', color='coral', ax=ax5)
    ax5.set_ylabel('Average Price')
    ax5.set_xlabel('Number of Stops')
    st.pyplot(fig5)
    st.markdown("**Insights:** Average flight price increases with the number of stops, with direct flights being the least expensive.")

# --- PAGE 4: Passenger SATISFACTION PREDICTOR ---
elif page == "Passenger Satisfaction Predictor":
    st.title("😊 Passenger Satisfaction Predictor")
    df_cust = load_customer_data()
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
        Class_encoded, Type_of_Travel_encoded, Gender_Male, Customer_Type_disloyal_Customer
    ]
    input_array = np.array(input_features).reshape(1, -1)
    if st.button("Predict Satisfaction"):
        pred_satisfaction = model.predict(input_array)
        label = "Satisfied" if pred_satisfaction[0] == 1 else "Not Satisfied"
        st.success(f"Predicted Passenger Satisfaction: {label}")

# --- PAGE 5: CUSTOMER SATISFACTION EDA (5 CHARTS) ---
elif page == "Passenger Satisfaction EDA":
    st.title("📊 Passenger Satisfaction EDA")
    df_cust = load_customer_data()
    sns.set_style("whitegrid")

    # 1. Mean Satisfaction Score by Class (Bar)
    st.subheader("1. Mean Satisfaction Score by Class (Bar)")
    if "Class_encoded" in df_cust.columns and "Customer_Satisfaction" in df_cust.columns:
       mean_by_class = df_cust.groupby('Class_encoded')["Customer_Satisfaction"].mean()
       fig1, ax1 = plt.subplots(figsize=(6,4))
       mean_by_class.plot(kind="bar", color="orange", ax=ax1)
       ax1.set_ylabel("Mean Satisfaction (Probability Satisfied)")
       ax1.set_xlabel("Travel Class (encoded)")
       st.pyplot(fig1)
       st.markdown("""
       **Legend:**  
       `0` = Business  
       `1` = Eco  
       `2` = Eco Plus
       """)
       st.markdown("**Insight:** Passengers in Business class have the highest satisfaction probability, while satisfaction is lowest in Eco Class.")
    else:
       st.info("Columns 'Class_encoded' or 'Customer_Satisfaction' not found.")

    # 2. Gender Distribution for Satisfaction (Countplot)
    st.subheader("2. Gender Distribution for Satisfaction (Countplot)")
    if "Gender_Male" in df_cust.columns and "Customer_Satisfaction" in df_cust.columns:
       df_cust['Gender'] = df_cust['Gender_Male'].map({0: 'Female', 1: 'Male'})
       fig2, ax2 = plt.subplots(figsize=(6,4))
       sns.countplot(x="Gender", hue="Customer_Satisfaction", data=df_cust, palette="pastel", ax=ax2)
       ax2.set_ylabel("Passenger Count")
       ax2.set_xlabel("Gender")
       ax2.legend(title="Satisfied", labels=["Not Satisfied", "Satisfied"])
       st.pyplot(fig2)
       st.markdown("**Insights:** Both males and females  have more 'Not Satisfied' passengers than 'satisfied', but the satisfaction proportions are similar across genders.")
    else:
       st.info("Columns 'Gender_Male' or 'Customer_Satisfaction' not found.")

    # 3. Age Distribution by Satisfaction (Boxplot)
    st.subheader("3. Age Distribution by Satisfaction (Boxplot)")
    if "Customer_Satisfaction" in df_cust.columns and "Age" in df_cust.columns:
       fig3, ax3 = plt.subplots(figsize=(7,4))
       sns.boxplot(x="Customer_Satisfaction", y="Age", data=df_cust, palette="Set3", ax=ax3)
       ax3.set_xlabel('Customer Satisfaction (0 = Not Satisfied, 1 = Satisfied)')
       ax3.set_ylabel('Age')
       st.pyplot(fig3)
       st.markdown("**Insight:** On average, older passengers report higher satisfaction than younger passengers. There are also a few outliers: some very elderly passengers are present in the satisfied group.")
    else:
       st.info("Columns 'Customer_Satisfaction' or 'Age' not found.")

    # 4. Flight Distance by Satisfaction (Boxplot)
    st.subheader("4. Flight Distance by Satisfaction (Boxplot)")
    if "Customer_Satisfaction" in df_cust.columns and "Flight Distance" in df_cust.columns:
       fig4, ax4 = plt.subplots(figsize=(7,4))
       sns.boxplot(x="Customer_Satisfaction", y="Flight Distance", data=df_cust, palette="coolwarm", ax=ax4)
       ax4.set_xlabel('Customer Satisfaction (0 = Not Satisfied, 1 = Satisfied)')
       ax4.set_ylabel('Flight Distance')
       st.pyplot(fig4)
       st.markdown("**Insight:** Passengers on longer flight tend to be more satisfied, and many short flights in the 'Not Satisfied' group show outliers with usually long distances. Most dissatisfaction is concentrated in shorter flights, but some long-haul passengers can still be dissatisfied.")
    else:
       st.info("Columns 'Customer_Satisfaction' or 'Flight Distance' not found.")

    # 5. Passenger Count by Travel Class (Countplot)
    st.subheader("5. Passenger Count by Travel Class (Countplot)")
    if "Class_encoded" in df_cust.columns:
       fig5, ax5 = plt.subplots(figsize=(6,4))
       sns.countplot(x="Class_encoded", data=df_cust, color="salmon", ax=ax5)
       ax5.set_ylabel("Passenger Count")
       ax5.set_xlabel("Travel Class (encoded)")
       st.pyplot(fig5)
       st.markdown("""
       **Legend:**  
       `0` = Business  
       `1` = Eco  
       `2` = Eco Plus
       """)
       st.markdown("**Insight:** Most passengers travel in Business and Eco classes, while Eco Plus class is chosen much less frequently.")
    else:
       st.info("Column 'Class_encoded' not found.")


