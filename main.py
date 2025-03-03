import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load preprocessor (OneHotEncoder, Scaler, etc.)
with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

# Define the expected input features
class CarFeatures(BaseModel):
    Levy: float
    Manufacturer: str
    Model: str
    Prod_year: int
    Category: str
    Leather_interior: str
    Fuel_type: str
    Engine_volume: float
    Mileage: float
    Cylinders: int
    Gear_box_type: str
    Drive_wheels: str
    Doors: int
    Wheel: str
    Color: str
    Airbags: int

app = FastAPI()

@app.post("/predict/")
def predict_price(features: CarFeatures):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # Apply preprocessing to match training data
        input_data_transformed = preprocessor.transform(input_data)

        # Ensure feature names match exactly
        correct_feature_names = list(preprocessor.get_feature_names_out())
        input_data_transformed = pd.DataFrame(input_data_transformed, columns=correct_feature_names)

        # Make prediction
        predicted_price = model.predict(input_data_transformed)[0]
        return {"predicted_price": predicted_price}

    except Exception as e:
        return {"error": str(e)}
