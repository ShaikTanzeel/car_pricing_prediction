"""
Car Price Prediction API
=========================
A FastAPI application that predicts car prices based on various features.

This API accepts car details (manufacturer, model, year, etc.) and returns
a predicted price using a trained machine learning model.

How it works:
1. User sends a POST request with car features
2. The preprocessor transforms the input (scales numeric, encodes categorical)
3. The trained model makes a prediction
4. The predicted price is returned

Author: Shaik Tanzeel Ahmed
"""

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ============================================================================
# Load the trained model and preprocessor
# ============================================================================
# These files were created by the train_model.py script
try:
    with open("car_price_model.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Run train_model.py first!")

try:
    with open("preprocessor.pkl", "rb") as file:
        preprocessor = pickle.load(file)
    print("Preprocessor loaded successfully!")
except FileNotFoundError:
    raise RuntimeError("Preprocessor file not found. Run train_model.py first!")

# ============================================================================
# Define the input schema using Pydantic
# ============================================================================
# This ensures the API receives the correct data types
class CarFeatures(BaseModel):
    """
    Input features for car price prediction.
    
    Each field corresponds to a characteristic of the car.
    The Field() function provides descriptions and examples for API documentation.
    """
    Levy: float = Field(
        ..., 
        description="Tax/levy amount in currency", 
        example=500.0
    )
    Manufacturer: str = Field(
        ..., 
        description="Car manufacturer/brand name", 
        example="TOYOTA"
    )
    Model: str = Field(
        ..., 
        description="Specific car model", 
        example="Corolla"
    )
    Prod_year: int = Field(
        ..., 
        description="Year of production", 
        example=2015
    )
    Category: str = Field(
        ..., 
        description="Car category/type", 
        example="Sedan"
    )
    Leather_interior: str = Field(
        ..., 
        description="Has leather interior? (Yes/No)", 
        example="Yes"
    )
    Fuel_type: str = Field(
        ..., 
        description="Type of fuel", 
        example="Petrol"
    )
    Engine_volume: float = Field(
        ..., 
        description="Engine volume in liters", 
        example=1.8
    )
    Mileage: float = Field(
        ..., 
        description="Total mileage in km", 
        example=50000.0
    )
    Cylinders: int = Field(
        ..., 
        description="Number of engine cylinders", 
        example=4
    )
    Gear_box_type: str = Field(
        ..., 
        description="Transmission type", 
        example="Automatic"
    )
    Drive_wheels: str = Field(
        ..., 
        description="Drive wheel type", 
        example="Front"
    )
    Doors: int = Field(
        ..., 
        description="Number of doors", 
        example=4
    )
    Wheel: str = Field(
        ..., 
        description="Steering wheel position", 
        example="Left wheel"
    )
    Color: str = Field(
        ..., 
        description="Car color", 
        example="Black"
    )
    Airbags: int = Field(
        ..., 
        description="Number of airbags", 
        example=6
    )
    
    class Config:
        """Configuration for the Pydantic model."""
        json_schema_extra = {
            "example": {
                "Levy": 500,
                "Manufacturer": "TOYOTA",
                "Model": "Corolla",
                "Prod_year": 2015,
                "Category": "Sedan",
                "Leather_interior": "Yes",
                "Fuel_type": "Petrol",
                "Engine_volume": 1.8,
                "Mileage": 50000,
                "Cylinders": 4,
                "Gear_box_type": "Automatic",
                "Drive_wheels": "Front",
                "Doors": 4,
                "Wheel": "Left wheel",
                "Color": "Black",
                "Airbags": 6
            }
        }


# ============================================================================
# Create the FastAPI application
# ============================================================================
app = FastAPI(
    title="Car Price Prediction API",
    description="""
    ## Overview
    This API predicts car prices based on various features like manufacturer, 
    model, year, mileage, and more.
    
    ## How to Use
    1. Send a POST request to `/predict/` with car features
    2. Receive predicted price in the response
    
    ## Model Information
    - Algorithm: Random Forest Regressor
    - Training Data: 19,237 car listings
    - Features: 16 input features
    - Accuracy: ~67% R² score
    """,
    version="1.0.0",
    contact={
        "name": "Shaik Tanzeel Ahmed",
        "url": "https://github.com/ShaikTanzeel/car_pricing_prediction"
    }
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Health Check"])
def root():
    """
    Health check endpoint.
    
    Returns a welcome message to confirm the API is running.
    """
    return {
        "message": "Welcome to the Car Price Prediction API!",
        "status": "online",
        "documentation": "Visit http://127.0.0.1:8000/docs for interactive API documentation.",
        "endpoints": {
            "root": "http://127.0.0.1:8000/",
            "health": "http://127.0.0.1:8000/health",
            "features": "http://127.0.0.1:8000/features",
            "predict": "http://127.0.0.1:8000/predict/ (POST request required)"
        }
    }


@app.get("/health", tags=["Health Check"])
def health_check():
    """
    Detailed health check endpoint.
    
    Returns the status of the model and preprocessor.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict/", tags=["Prediction"])
@app.post("/predict", tags=["Prediction"])
def predict_price(features: CarFeatures):
    """
    Predict car price based on input features.
    
    ## Input
    Send car details in the request body following the CarFeatures schema.
    
    ## Output
    Returns the predicted price in USD.
    
    ## Example Response
    ```json
    {
        "predicted_price": 21402.25,
        "currency": "USD",
        "status": "success"
    }
    ```
    """
    try:
        # Step 1: Convert input to DataFrame
        # The model expects a pandas DataFrame, not a dictionary
        input_dict = features.model_dump()  # Pydantic v2 method
        input_data = pd.DataFrame([input_dict])
        
        # Step 2: Apply preprocessing
        # The ColumnTransformer handles:
        # - Scaling numeric features (StandardScaler)
        # - Encoding categorical features (OneHotEncoder)
        input_processed = preprocessor.transform(input_data)
        
        # Step 3: Make prediction
        predicted_price = model.predict(input_processed)[0]
        
        # Step 4: Return result
        return {
            "predicted_price": round(float(predicted_price), 2),
            "currency": "USD",
            "status": "success"
        }
        
    except Exception as e:
        # Return detailed error for debugging
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Prediction failed. Please check your input data.",
                "status": "error"
            }
        )


@app.get("/predict/", tags=["Prediction"], include_in_schema=False)
@app.get("/predict", tags=["Prediction"], include_in_schema=False)
def predict_get_guide():
    """
    Guides users who try to access the prediction endpoint via GET.
    """
    return {
        "message": "This endpoint requires a POST request with car features.",
        "hint": "Use the /docs page to test the API interactively.",
        "docs_url": "/docs",
        "method_required": "POST"
    }


@app.get("/features", tags=["Information"])
def get_features():
    """
    Get information about expected input features.
    
    Returns details about each feature the model expects.
    """
    return {
        "features": {
            "Levy": "Tax/levy amount (numeric, e.g., 500)",
            "Manufacturer": "Car brand (e.g., TOYOTA, BMW, MERCEDES-BENZ)",
            "Model": "Car model (e.g., Corolla, 3 Series)",
            "Prod_year": "Production year (e.g., 2015)",
            "Category": "Car type (e.g., Sedan, SUV, Hatchback)",
            "Leather_interior": "Has leather interior (Yes/No)",
            "Fuel_type": "Fuel type (Petrol, Diesel, Hybrid, etc.)",
            "Engine_volume": "Engine size in liters (e.g., 1.8)",
            "Mileage": "Total distance driven in km (e.g., 50000)",
            "Cylinders": "Engine cylinders (e.g., 4, 6, 8)",
            "Gear_box_type": "Transmission (Automatic, Manual, Tiptronic)",
            "Drive_wheels": "Drive type (Front, Rear, 4x4)",
            "Doors": "Number of doors (2, 4, 5)",
            "Wheel": "Steering position (Left wheel, Right-hand drive)",
            "Color": "Car color (e.g., Black, White, Silver)",
            "Airbags": "Number of airbags (e.g., 6, 8, 12)"
        },
        "total_features": 16
    }


# ============================================================================
# Run the application (for local development)
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
