import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# Check for conflicting code.py file
if os.path.exists('code.py'):
    st.error("A file named 'code.py' was found, which conflicts with Python's standard library. Please rename or remove 'code.py' and restart.")
    st.stop()

# Streamlit app configuration
st.set_page_config(page_title="Simple Stock Prediction", layout="wide")
st.title("📈 Simple Stock Price Prediction")
st.markdown("Choose a stock, date range, and model to see predictions. Perfect for beginners!")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.selectbox("Stock Symbol", ["^GSPC", "AAPL", "MSFT", "GOOGL", "TSLA"], index=0)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"), min_value=pd.to_datetime("2000-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"), max_value=pd.to_datetime("2025-07-22"))
    model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"], index=0)
    run_button = st.button("Run Analysis")

def load_data(symbol, start_date, end_date):
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found. Try a different symbol or date range.")
            return None
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Missing columns. Available: {list(data.columns)}")
            return None
        st.success(f"Loaded {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocess data for modeling"""
    if data is None:
        return None, None, None
    df = data.copy()
    scaler = MinMaxScaler()
    features = ['High', 'Low', 'Close', 'Volume']
    df[features] = scaler.fit_transform(df[features])
    df['target'] = df['Open'].shift(-1)
    df = df.dropna()
    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    return df, train_data, test_data

def build_model(model_name, train_data, test_data):
    """Build and train the selected model"""
    X_train = train_data[['High', 'Low', 'Close', 'Volume']]
    y_train = train_data['target']
    X_test = test_data[['High', 'Low', 'Close', 'Volume']]
    y_test = test_data['target']
    
    model = LinearRegression() if model_name == "Linear Regression" else RandomForestRegressor(random_state=42, n_estimators=50)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    results = {
        model_name: {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_pred': train_pred,
            'test_pred': test_pred
        }
    }
    return results

def display_descriptive_stats(data):
    """Display descriptive statistics"""
    if data is not None:
        st.subheader("📊 Stock Data Summary")
        desc = data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        st.dataframe(
            desc.style.format("{:.2f}").set_properties(**{
                'background-color': '#e6f3ff',
                'border-color': '#4b0082',
                'text-align': 'center',
                'font-size': '14px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#4b0082'), ('color', 'white'), ('font-weight', 'bold')]}
            ])
        )

def plot_predictions(test_data, results, model_name):
    """Plot actual vs predicted prices"""
    st.subheader(f"📈 {model_name} - Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data['target'], mode='lines', name='Actual',
        line=dict(color='navy', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=test_data.index, y=results[model_name]['test_pred'], mode='lines', name='Predicted',
        line=dict(color='crimson', width=2, dash='dash')
    ))
    fig.update_layout(
        title=f"{model_name} Predictions",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_white",
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

def display_results(results, model_name):
    """Display model results"""
    st.subheader(f"📋 {model_name} Results")
    df = pd.DataFrame({
        'Metric': ['Train RMSE', 'Test RMSE'],
        'Value': [
            f"{results[model_name]['train_rmse']:.2f}",
            f"{results[model_name]['test_rmse']:.2f}"
        ]
    })
    st.dataframe(
        df.style.set_properties(**{
            'background-color': '#e6f3ff',
            'border-color': '#4b0082',
            'text-align': 'center',
            'font-size': '14px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#4b0082'), ('color', 'white'), ('font-weight', 'bold')]}
        ])
    )

if run_button:
    with st.spinner("Running analysis..."):
        raw_data = load_data(symbol, start_date, end_date)
        if raw_data is not None:
            display_descriptive_stats(raw_data)
            processed_data, train_data, test_data = preprocess_data(raw_data)
            if processed_data is not None:
                results = build_model(model_choice, train_data, test_data)
                display_results(results, model_choice)
                plot_predictions(test_data, results, model_choice)