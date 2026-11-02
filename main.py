
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import HTTPException



app = FastAPI(title="Play Tennis Prediction API")



model_data = joblib.load("tennis_model.pkl")

final_model = model_data["model"]

weather_map = model_data["weather_map"]
temp_map = model_data["temp_map"]
humidity_map = model_data["humidity_map"]
wind_map = model_data["wind_map"]



class TennisData(BaseModel):

    outlook: Literal["Sunny", "Overcast", "Rain"]

    temp: Literal["Hot", "Mild", "Cool"]

    humidity: Literal["High", "Normal"]

    wind: Literal["Weak", "Strong"]



@app.get("/")
def home():

    return {
        "message": "Play Tennis Prediction API is running!"
    }



@app.post("/predict")
def predict(data: TennisData):


    query = pd.DataFrame([{
        "outlook": weather_map[data.outlook],
        "temp": temp_map[data.temp],
        "humidity": humidity_map[data.humidity],
        "wind": wind_map[data.wind]
    }])


    prediction = final_model.predict(query)

    probabilities = final_model.predict_proba(query)


    if prediction[0] == 1:

        result = "Yes, Play Tennis"

    else:

        result = "No, Do Not Play Tennis"


    return {
        "prediction": int(prediction[0]),
        "result": result,
    }
