**Car Price Prediction API**

**Overview**

This project builds a Car Price Prediction API using FastAPI, which allows users to input car features and receive a predicted price. The machine learning model is trained on real-world car pricing data and is deployed on Render.

**Features**
Predict car prices based on input features
FastAPI framework for efficient API handling
Deployed on Render for online accessibility
GitHub integration for version control


**Project Structure**

car_pricing_prediction/
│── main.py             # FastAPI application
│── requirements.txt    # Dependencies
│── car_price_model.pkl # Trained ML model
│── README.md           # Project documentation
│── .gitignore          # Ignored files
└── data/               # Dataset (if applicable)

**Installation & Setup**

1. Clone the repository

git clone https://github.com/ShaikTanzeel/car_pricing_prediction.git
cd car_pricing_prediction

2. Create a virtual environment

python -m venv apienv
source apienv/bin/activate  # On Windows: apienv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

**Running Locally**

1. Start FastAPI server

uvicorn main:app --reload

2. Access API documentation

Open Swagger UI: http://127.0.0.1:8000/docs

Open Redoc UI: http://127.0.0.1:8000/redoc

API Endpoints

1. Home Endpoint

GET /

{
  "message": "Welcome to FastAPI!"
}

2. Prediction Endpoint

POST /predict/

Request Body (JSON):

{
  "Levy": 0,
  "Manufacturer": "Toyota",
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
  "Wheel": "Left",
  "Color": "Black",
  "Airbags": 6
}

Response:

{
  "predicted_price": 12000.50
}

Deployment

The API is deployed on Render and can be accessed at: 🔗 Live API on Render (Replace with actual link)

Version Control

GitHub Repository: https://github.com/ShaikTanzeel/car_pricing_prediction

Git Commands Used:

git add .
git commit -m "Updated FastAPI script"
git push origin main

Contributors
Shaik Tanzeel Ahmed 