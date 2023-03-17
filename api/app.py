import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow


description = """

Welcome to Getaround API. This API uses a state-of-the-art machine learning model to help you determine the optimum car rental price based on the characteristics of the car.


"""

tag_metadata = [
    {
        "name": "Index",
        "description": "Default endpoint"
    },

    {
        "name": "Price Prediction",
        "description": "Get a suggestion of car rental price per day based on the characteristics of the car"
    }
]

# Creating an instance of FastAPI class
app = FastAPI(
    title="Getaround API",
    description = description,
    openapi_tags=tag_metadata
    )


class PredictionFeatures(BaseModel):
    model_key: str = "Citroën"
    mileage: int = 150000
    engine_power: int = 135
    fuel: str = "diesel"
    paint_color: str = "black"
    car_type: str = "estate"
    private_parking_available: bool = True
    has_gps: bool = True
    has_air_conditioning: bool = False
    automatic_car: bool = False
    has_getaround_connect: bool = False
    has_speed_regulator: bool = False
    winter_tires: bool = True

# Defining endpoints
@app.get("/", tags=["Index"])
async def index():
    message = "Welcome to the Getaround API!"
    return message


@app.post("/predict", tags = ["Price Prediction"])
async def predict(features: PredictionFeatures):

    data = pd.DataFrame({"model_key": [features.model_key],
                         "mileage": [features.mileage],
                         "engine_power": [features.engine_power],
                         "fuel": [features.fuel],
                         "paint_color": [features.paint_color],
                         "car_type": [features.car_type],
                         "private_parking_available": [features.private_parking_available],
                         "has_gps": [features.has_gps],
                         "has_air_conditioning": [features.has_air_conditioning],
                         "automatic_car": [features.automatic_car],
                         "has_getaround_connect": [features.has_getaround_connect],
                         "has_speed_regulator": [features.has_speed_regulator],
                         "winter_tires": [features.winter_tires]}, index=[0])

    logged_model = 'runs:/106dde20204f451a8bc6faf5adbae154/getaround'

    # Loading model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predicting on a Pandas DataFrame.
    prediction = loaded_model.predict(pd.DataFrame(data))

    # Formatting response
    response = {
        "prediction": prediction.tolist()[0]
    }

    return response


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000) 
    # Defining web server to run the `app` variable (which contains FastAPI instance), 
    # with a specific host IP (0.0.0.0) and port (4000)