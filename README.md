# Flight & Customer Prediction

A machine learning project for predicting flight prices and customer satisfaction using Python, scikit-learn, XGBoost, and Streamlit.

---

## Project Structure

├── mlartifacts/ # MLflow and model artifacts
├── mlruns/ # MLflow experiment logs
├── Flight_Price data.csv # Main dataset for flight price prediction
├── Passenger_Satisfaction_data.csv # Dataset for customer satisfaction prediction
├── FPStreamlitApp.py # Streamlit app for flight price prediction
├── CSStreamlitApp.py # Streamlit app for customer satisfaction
├── scripts.ipynb # Jupyter notebook for exploration/training
├── scripts2.ipynb # Additional scripts/notebooks
├── models.pkl # Pickled multiple trained pipelines
├── rf_model.pkl # Pickled Random Forest model
├── mlflow.db # MLflow tracking DB

---

## Features

- **Flight Price Prediction:** Predicts ticket prices based on origin, destination, airline, journey details, etc.
- **Customer Satisfaction Prediction:** Analyzes factors influencing customer satisfaction.
- **Multiple Model Pipelines:** Linear Regression, Random Forest, XGBoost pipelines with preprocessing.
- **Experiment Tracking:** Full MLflow integration for model/metric comparison.
- **Interactive Web Apps:** User-friendly Streamlit apps for real-time predictions.

---

## Setup & Usage

### 1. Clone the Repository


### 2. Install Requirements

> **Note:** Create a `requirements.txt` with all necessary packages (e.g., pandas, scikit-learn, xgboost, streamlit, mlflow).

---

### 3. Training Models

- Open `scripts.ipynb` in Jupyter Notebook or VSCode.
- Follow the code to preprocess data, train, evaluate, and export models to `.pkl` files.

---

### 4. Running the Streamlit Apps

For Flight Price Prediction:
For Customer Satisfaction Prediction:
Use the web UI to select models and enter feature values for prediction.

---

### 5. Experiment Tracking with MLflow (Optional)

Start MLflow UI locally:
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Model Details

- **Preprocessing:** Uses `ColumnTransformer` for categorical (OneHotEncoding) and numerical (StandardScaler) features.
- **Models:** Includes Linear Regression, Random Forest, XGBoost (easily extendable).
- **Model Serialization:** Models and pipelines are saved as pickle files for deployment/inference.

---

## Data

- **Flight_Price data.csv:** Features like Source, Destination, Airline, Duration, Stops, Journey info, etc.
- **Passenger_Satisfaction_data.csv:** Inputs for customer satisfaction prediction.

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/feature-name`)
3. Commit your changes (`git commit -m 'add some feature'`)
4. Push to the branch (`git push origin feature/feature-name`)
5. Open a pull request

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [Streamlit](https://streamlit.io/)
- [MLflow](https://mlflow.org/)

---


