# main.py
from fastapi import FastAPI
from datetime import datetime
import joblib
import json
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import HTTPException
from datetime import datetime
import requests
from requests.exceptions import Timeout, ConnectionError

app = FastAPI(
    title="Weather Intelligence API",
    description="AI-powered weather prediction service for Sydney, Australia",
    version="1.0.0"
)

# Load models
cci_model_T1 = joblib.load("cci_model_T1.pkl")
cci_model_T2 = joblib.load("cci_model_T2.pkl")
cci_model_T3 = joblib.load("cci_model_T3.pkl")
whc_model = joblib.load("whc_model.pkl")

# Load feature cols
with open("cci_feature_cols.json", "r") as f:
    cci_feature_cols = json.load(f)

with open("whc_feature_cols.json", "r") as f:
    whc_feature_cols = json.load(f)



cci_feature_importance = dict(
    sorted(
        zip(cci_feature_cols, cci_model_T1.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
)

whc_feature_importance = dict(
    sorted(
        zip(whc_feature_cols, whc_model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True
    )



# GET /
@app.get("/")
def root():
    return {
        "message": "Welcome to the Weather Intelligence API",
        "description": "AI-powered weather prediction service for Sydney, Australia",
        "endpoints": {
            "health": "/health",
            "cci_prediction": "/predict/index/comfort_climate",
            "whc_prediction": "/predict/category/weather_hazard",
            "model_metadata": "/model-metadata"
        }
    }


# GET /health
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Weather Intelligence API is running",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "cci_T1": cci_model_T1 is not None,
            "cci_T2": cci_model_T2 is not None,
            "cci_T3": cci_model_T3 is not None,
            "whc": whc_model is not None
        }
    }



# GET /model-metadata
@app.get("/model-metadata")
def model_metadata():
    return [
        {
            "target": "Climate Comfort Index (CCI)",
            "prediction_type": "Regression",
            "algorithm": "XGBoost Regressor",
            "forecast_horizon": "T+1, T+2, T+3 (1-3 days ahead)",
            "features": cci_feature_cols,
            "feature_importance": cci_feature_importance,  
            "model_performance": {
                "T+1_MAE": 6.33,
                "T+2_MAE": 7.57,
                "T+3_MAE": 7.70
            }
        },
        {
            "target": "Weather Hazard Category (WHC)",
            "prediction_type": "Multiclass Classification",
            "algorithm": "Random Forest Classifier",
            "forecast_horizon": "T+7 (exactly 7 days ahead)",
            "features": whc_feature_cols,
            "feature_importance": whc_feature_importance, 
            "classes": ["Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"],
            "model_performance": {
                "F1_Macro": 0.2888,
                "Accuracy": 0.4726,
                "High_Risk_Recall": 0.35
            }
        }
    ]



class DateInput(BaseModel):
    date: str  # format: YYYY-MM-DD




# check date
def validate_date(date_str: str):
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Please use YYYY-MM-DD format (e.g. 2025-01-01)."
        )
    
    # check date, can't be today or furture day
    if date >= datetime.now():
        raise HTTPException(
            status_code=400,
            detail=f"Date must be in the past. Please provide a date before today ({datetime.now().strftime('%Y-%m-%d')})."
        )
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    if date < datetime(2010, 1, 15):
        raise HTTPException(
            status_code=400,
            detail=f"Date out of range. The available data spans from 2010-01-15 to {yesterday}. Please provide a date within this range."
        )
    
    return date




def fetch_weather_data(end_date: str):
    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end - timedelta(days=30)
    
    params = {
        "latitude": -33.8688,
        "longitude": 151.2093,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "apparent_temperature_mean", "relative_humidity_2m_mean",
            "wind_speed_10m_mean", "wind_direction_10m_dominant",
            "cloud_cover_mean", "precipitation_sum",
            "wind_gusts_10m_max", "snowfall_sum", "sunshine_duration",
            "daylight_duration"
        ],
        "timezone": "Australia/Sydney"
    }
    
    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive", 
            params=params,
            timeout=10
        )
        response.raise_for_status()
    except Timeout:
        raise HTTPException(
            status_code=504,
            detail="Request to Open-Meteo API timed out. Please try again later."
        )
    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Failed to connect to Open-Meteo API. Please check your internet connection and try again."
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch weather data from Open-Meteo API. Please try again later."
        )
    
    df = pd.DataFrame(response.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df


@app.post("/predict/index/comfort_climate")
def predict_cci(input: DateInput):
    validate_date(input.date)
    # get data
    df = fetch_weather_data(input.date)
    
    # compute lag features
    raw_cols = [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "apparent_temperature_mean", "relative_humidity_2m_mean",
        "wind_speed_10m_mean", "wind_direction_10m_dominant",  
        "cloud_cover_mean", "precipitation_sum",
        "wind_gusts_10m_max", "snowfall_sum", "sunshine_duration",
        "daylight_duration"
    ]
    
    for col in raw_cols:
        for lag in [1, 2, 3, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    for col in raw_cols:
        for window in [7, 14]:
            df[f"{col}_roll{window}_mean"] = df[col].shift(1).rolling(window).mean()

    # Streak features
    def calc_cci(row):
        temp_score = max(0, 1 - abs(row["temperature_2m_mean"] - 22) / 20)
        humidity_score = max(0, 1 - abs(row["relative_humidity_2m_mean"] - 50) / 50)
        wind_score = max(0, 1 - abs(row["wind_speed_10m_mean"] - 10) / 40)
        cloud_score = max(0, 1 - row["cloud_cover_mean"] / 100)
        rain_score = max(0, 1 - row["precipitation_sum"] / 20)
        return 100 * (0.35 * temp_score + 0.20 * humidity_score + 
                      0.15 * wind_score + 0.15 * cloud_score + 0.15 * rain_score)

    df["CCI"] = df.apply(calc_cci, axis=1)
    
    good_weather = df["CCI"] > 70
    df["good_weather_streak"] = (
        good_weather.groupby((~good_weather).cumsum()).cumcount()
    )
    bad_weather = df["CCI"] < 50
    df["bad_weather_streak"] = (
        bad_weather.groupby((~bad_weather).cumsum()).cumcount()
    )
    rainy = df["precipitation_sum"] > 0
    df["rain_streak"] = (
        rainy.groupby((~rainy).cumsum()).cumcount()
    )
    
    # get last row
    last_row = df.iloc[-1][cci_feature_cols].values.reshape(1, -1)

    
    date = datetime.strptime(input.date, "%Y-%m-%d")
    return {
        "input_date": input.date,
        "predictions": {
            "comfort_climate": {
                (date + timedelta(days=1)).strftime("%Y-%m-%d"): round(float(cci_model_T1.predict(last_row)[0]), 2),
                (date + timedelta(days=2)).strftime("%Y-%m-%d"): round(float(cci_model_T2.predict(last_row)[0]), 2),
                (date + timedelta(days=3)).strftime("%Y-%m-%d"): round(float(cci_model_T3.predict(last_row)[0]), 2)
            }
        }
    }




@app.post("/predict/category/weather_hazard")
def predict_whc(input: DateInput):
    validate_date(input.date)
    # get data
    end = datetime.strptime(input.date, "%Y-%m-%d")
    start = end - timedelta(days=30)
    
    params = {
        "latitude": -33.8688,
        "longitude": 151.2093,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": input.date,
        "daily": [
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "apparent_temperature_mean", "relative_humidity_2m_mean",
            "wind_speed_10m_mean", "wind_direction_10m_dominant",
            "cloud_cover_mean", "precipitation_sum", "wind_gusts_10m_max",
            "snowfall_sum", "sunshine_duration", "daylight_duration",
            "rain_sum", "precipitation_hours", "wind_speed_10m_max",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration",
            "apparent_temperature_max", "apparent_temperature_min"
        ],
        "timezone": "Australia/Sydney"
    }
    
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    df = pd.DataFrame(response.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])

    # Feature engineering
    raw_cols_whc = [col for col in df.columns if col != "time"]
    
    for col in raw_cols_whc:
        for lag in [1, 2, 3, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    for col in raw_cols_whc:
        for window in [7, 14]:
            df[f"{col}_roll{window}_mean"] = df[col].shift(1).rolling(window).mean()

    # Binary threshold indicators
    df["heavy_rain"] = (df["precipitation_sum"] > 30).astype(int)
    df["strong_wind"] = (df["wind_gusts_10m_max"] > 80).astype(int)
    df["heavy_rain_and_wind"] = (
        (df["precipitation_sum"] > 30) & 
        (df["wind_gusts_10m_max"] > 80)
    ).astype(int)

    # 取最後一筆
    last_row = df.iloc[-1][whc_feature_cols].values.reshape(1, -1)
    
    # 預測
    prediction = int(whc_model.predict(last_row)[0])
    
    whc_labels = {
        0: "Low Risk",
        1: "Moderate Risk", 
        2: "High Risk",
        3: "Extreme Risk"
    }
    
    return {
        "input_date": input.date,
        "predictions": {
            "weather_hazard": {
                (end + timedelta(days=7)).strftime("%Y-%m-%d"): whc_labels[prediction]
            }
        }
    }
