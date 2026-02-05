# Car Price Prediction API

A FastAPI-based machine learning API that predicts car prices based on various features.

## Overview

This project uses a **Random Forest Regressor** trained on 19,237 car listings to predict prices based on features like manufacturer, model, year, mileage, and more.

**Model Performance:**
- R² Score: 67.26%
- Mean Absolute Error: $5,113

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (if needed)
```bash
# Download dataset (requires Kaggle API)
kaggle datasets download deepcontractor/car-price-prediction-challenge -p data --unzip

# Train model
python train_model.py
```

### 3. Run the API
```bash
uvicorn main:app --reload
```

### 4. Test the API
Open http://127.0.0.1:8000/docs for interactive documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed status |
| GET | `/features` | Feature information |
| POST | `/predict/` | Price prediction |

## Example Request

```json
POST /predict/
{
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
```

**Response:**
```json
{
  "predicted_price": 21402.25,
  "currency": "USD",
  "status": "success"
}
```

## Project Structure

```
car_pricing_prediction/
├── main.py              # FastAPI application
├── train_model.py       # Model training script
├── car_price_model.pkl  # Trained model
├── preprocessor.pkl     # Data preprocessor
├── requirements.txt     # Dependencies
├── PROJECT_GUIDE.md     # Detailed learning guide
├── README.md            # This file
└── data/                # Training dataset
```

## Documentation

For a detailed explanation of the project, including:
- What was wrong and how it was fixed
- Machine learning concepts explained
- Step-by-step code walkthrough

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md)

## Author

**Shaik Tanzeel Ahmed**

## License

MIT License