import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load the trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define request model
class CarFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float
    feature14: float
    feature15: float
    feature16: float
    feature17: float
    feature18: float


# Define home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to FastAPI!"}

# Define prediction endpoint
@app.post("/predict/")
def predict_price(features: CarFeatures):
    input_data = np.array([[features.feature1, features.feature2, features.feature3, features.feature4, features.feature5]])
    predicted_price = model.predict(input_data)[0]
    return {"predicted_price": predicted_price}

# Run the API (only for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
