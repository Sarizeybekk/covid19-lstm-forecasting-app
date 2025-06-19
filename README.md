# COVID-19 LSTM Forecasting Project

## Project Description

This project forecasts daily new COVID-19 cases using an LSTM (Long Short-Term Memory) deep learning model applied to time series data. It includes both model training and an interactive Streamlit application for real-time forecasting. The system is fully automated after initial model training.

## Technologies Used

- Python 3
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Streamlit

## The Notebook Will

- Train a single-layer LSTM model.
- Evaluate the model's performance.
- Save the trained model to `covid_lstm_cnn_model.h5`.
- Save scaler parameters to `scaler_min.npy` and `scaler_scale.npy`.

## Running the Streamlit Application

Once the model is trained and saved, launch the Streamlit web app:

```bash
streamlit run app.py
```
## The Application Will Automatically

- Load the dataset and the trained model.
- Forecast the next 7 days of new COVID-19 cases.
- Display both numeric results and forecast plots.
- No file uploads are required; the system is fully automated.

## Installation

Install the required packages:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib streamlit
```
## Forecast Model Details

- **Input Sequence**: Last 10 days of "New cases".
- **Model Architecture**: Single-layer LSTM with 64 units and Dense output.
- **Loss Function**: Mean Squared Error (MSE).
- **Forecast Horizon**: 7 days ahead.
- **Forecast Type**: Recursive forecasting.

## Future Improvements

- Incorporate additional external features (lockdown measures, mobility data, holidays, vaccination rates).
- Develop hybrid CNN-LSTM model architecture for better pattern recognition.
- Apply hyperparameter optimization using automated search frameworks.
- Extend forecast horizon beyond 7 days.
- Deploy using Docker or cloud platforms for production environments.



