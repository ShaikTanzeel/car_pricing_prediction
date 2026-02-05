# Car Price Prediction Project - Complete Learning Guide

> **Author**: Shaik Tanzeel Ahmed  
> **GitHub**: [ShaikTanzeel/car_pricing_prediction](https://github.com/ShaikTanzeel/car_pricing_prediction)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What Was Wrong (Original Issues)](#what-was-wrong)
3. [How It Was Fixed](#how-it-was-fixed)
4. [Project Architecture](#project-architecture)
5. [Understanding the Code](#understanding-the-code)
6. [Key Machine Learning Concepts](#key-ml-concepts)
7. [How to Run the Project](#how-to-run)
8. [API Documentation](#api-documentation)
9. [Files Changed](#files-changed)
10. [What I Learned](#what-i-learned)

---

## Project Overview

This project builds a **Car Price Prediction API** using:
- **FastAPI** - A modern Python web framework for building APIs
- **scikit-learn** - Machine learning library for model training
- **pandas** - Data manipulation and analysis
- **Random Forest Regressor** - The ML algorithm used for prediction

**What it does**: Takes car details (manufacturer, model, year, mileage, etc.) and predicts the price.

---

## What Was Wrong (Original Issues)

### Issue 1: Preprocessor-Model Mismatch

**Problem**: The `preprocessor.pkl` was a simple `StandardScaler` but the model expected 6,737 features.

```
# What was saved:
StandardScaler()  # Only scales numbers

# What the model needed:
ColumnTransformer with OneHotEncoder  # Converts categories to numbers
```

**Why this is a problem**: The model was trained on data that was one-hot encoded (converted categories to binary columns), but the preprocessor couldn't do this transformation.

### Issue 2: Column Naming Mismatch

**Problem**: The API and model expected different column names.

| API Expected | Model Trained With |
|-------------|-------------------|
| `Prod_year` | `Prod. year` |
| `Engine_volume` | `Engine volume` |
| `Fuel_type` | `Fuel type` |

**Why this is a problem**: When column names don't match, the preprocessor can't find the data.

### Issue 3: Incorrect Training Pipeline

**Problem**: The original training encoded ALL columns as categories, even numeric ones.

```python
# Wrong approach (what was done):
Mileage → "50000 km" → One-hot encoded → Mileage_50000_km = 1

# Correct approach (what should be done):
Mileage → 50000 → Scaled → -0.234 (standardized value)
```

**Why this is a problem**: 
- Created 6,737 features instead of ~1,500
- Mileage of 50,001 km would be "unknown" because only exact values seen in training work
- Loses the numeric meaning of the data

---

## How It Was Fixed

### Fix 1: Created Proper Training Script (`train_model.py`)

```python
# Properly separate numeric and categorical features
numeric_features = ['Levy', 'Prod_year', 'Engine_volume', 'Mileage', 
                    'Cylinders', 'Doors', 'Airbags']
categorical_features = ['Manufacturer', 'Model', 'Category', ...]

# Use ColumnTransformer to handle both types
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),      # Scale numbers
    ('cat', OneHotEncoder(), categorical_features)    # Encode categories
])
```

### Fix 2: Cleaned Column Names

```python
# Renamed columns to match API expectations
column_mapping = {
    'Prod. year': 'Prod_year',
    'Leather interior': 'Leather_interior',
    'Fuel type': 'Fuel_type',
    'Engine volume': 'Engine_volume',
    'Gear box type': 'Gear_box_type',
    'Drive wheels': 'Drive_wheels'
}
df = df.rename(columns=column_mapping)
```

### Fix 3: Proper Data Cleaning

```python
# Clean Mileage (remove ' km' suffix)
df['Mileage'] = df['Mileage'].str.replace(' km', '')
df['Mileage'] = pd.to_numeric(df['Mileage'])

# Clean Engine volume (extract number from "2.0 Turbo")
df['Engine volume'] = df['Engine volume'].str.extract(r'(\d+\.?\d*)')[0]
```

---

## Project Architecture

```
car_pricing_prediction/
│
├── main.py                 # FastAPI application (the API)
├── train_model.py          # Training script (creates the model)
├── car_price_model.pkl     # Trained ML model (binary file)
├── preprocessor.pkl        # Data transformer (binary file)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore             # Git ignore rules
│
└── data/
    └── car_price_prediction.csv  # Training dataset (19,237 cars)
```

### How the Parts Connect

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
│  (Run once to create model)                                 │
│                                                             │
│  data/car_price_prediction.csv                              │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ train_model.py  │                                        │
│  │                 │                                        │
│  │ 1. Load data    │                                        │
│  │ 2. Clean data   │                                        │
│  │ 3. Split data   │                                        │
│  │ 4. Fit preproc  │                                        │
│  │ 5. Train model  │                                        │
│  │ 6. Save files   │                                        │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐   ┌─────────────────┐                  │
│  │ preprocessor.pkl│   │car_price_model.pkl│                │
│  │ (transformer)   │   │ (trained model) │                  │
│  └─────────────────┘   └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION PHASE                         │
│  (API handles requests)                                     │
│                                                             │
│  User Request (JSON)                                        │
│  {"Manufacturer": "TOYOTA", "Model": "Corolla", ...}        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────┐        │
│  │               main.py (FastAPI)                  │       │
│  │                                                  │       │
│  │  1. Receive request                              │       │
│  │  2. Validate with Pydantic                       │       │
│  │  3. Convert to DataFrame                         │       │
│  │  4. Transform with preprocessor.pkl              │       │
│  │  5. Predict with car_price_model.pkl             │       │
│  │  6. Return JSON response                         │       │
│  └─────────────────────────────────────────────────┘        │
│           │                                                 │
│           ▼                                                 │
│  Response: {"predicted_price": 21402.25}                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Understanding the Code

### 1. train_model.py - The Training Script

This script does everything needed to create a working ML model.

#### Step 1: Load Data
```python
df = pd.read_csv('data/car_price_prediction.csv')
```
- Reads the CSV file with 19,237 car listings
- Each row is one car with features and price

#### Step 2: Data Cleaning
```python
# Handle missing values
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
df['Levy'] = df['Levy'].fillna(0)

# Clean string columns
df['Mileage'] = df['Mileage'].str.replace(' km', '')
```
- `errors='coerce'`: Convert invalid values to NaN instead of crashing
- `fillna(0)`: Replace missing values with 0

#### Step 3: Define Features and Target
```python
# What we want to predict (target)
y = df['Price']

# What we use to predict (features)
X = df.drop(columns=['Price', 'ID'])
```
- **Target (y)**: The price we want to predict
- **Features (X)**: All other columns that help predict

#### Step 4: Split Data
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- **80%** for training the model
- **20%** for testing (model never sees this during training)
- `random_state=42`: Makes the split reproducible

#### Step 5: Create Preprocessor
```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
```

**StandardScaler**: Normalizes numbers to have mean=0 and std=1
```
Before: [50000, 100000, 25000]
After:  [-0.23, 1.12, -0.89]
```

**OneHotEncoder**: Converts categories to binary columns
```
Before: Color = "Black"
After:  Color_Black=1, Color_White=0, Color_Red=0, ...
```

#### Step 6: Train Model
```python
model = RandomForestRegressor(
    n_estimators=100,    # 100 decision trees
    max_depth=15,        # Maximum depth of each tree
    random_state=42
)
model.fit(X_train_processed, y_train)
```

**Random Forest**: An ensemble of decision trees
- Each tree votes on the prediction
- Final prediction = average of all trees
- More robust than a single decision tree

#### Step 7: Evaluate
```python
y_pred = model.predict(X_test_processed)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**Metrics**:
- **MAE (Mean Absolute Error)**: Average $ difference between prediction and actual
- **R² Score**: Percentage of variance explained (0.67 = 67% accurate)

---

### 2. main.py - The API

#### FastAPI App Setup
```python
app = FastAPI(
    title="Car Price Prediction API",
    description="...",
    version="1.0.0"
)
```

#### Pydantic Model (Input Validation)
```python
class CarFeatures(BaseModel):
    Levy: float
    Manufacturer: str
    # ... etc
```
- **Automatically validates** incoming data
- **Generates documentation** for the API
- **Returns clear errors** if data is wrong

#### Prediction Endpoint
```python
@app.post("/predict/")
def predict_price(features: CarFeatures):
    # 1. Convert to DataFrame
    input_data = pd.DataFrame([features.model_dump()])
    
    # 2. Transform
    input_processed = preprocessor.transform(input_data)
    
    # 3. Predict
    price = model.predict(input_processed)[0]
    
    return {"predicted_price": price}
```

---

## Key ML Concepts

### 1. Feature Types

| Type | Examples | Preprocessing |
|------|----------|---------------|
| Numeric | Mileage, Year, Engine Size | StandardScaler |
| Categorical | Manufacturer, Color | OneHotEncoder |

### 2. Why We Scale Numbers

Neural networks and some algorithms work better when numbers are on similar scales.

```
Without scaling:        With scaling:
Mileage: 150,000       Mileage: 0.8
Year: 2015             Year: -0.2
Airbags: 6             Airbags: 0.5
```

### 3. Why We One-Hot Encode Categories

ML models can't understand text directly. We convert to numbers:

```
Color = ["Red", "Blue", "Green"]

Becomes:
Color_Red   Color_Blue   Color_Green
    1           0            0       (for a red car)
    0           1            0       (for a blue car)
```

### 4. Random Forest Algorithm

```
                    Data
                      │
        ┌──────┬──────┼──────┬──────┐
        ▼      ▼      ▼      ▼      ▼
      Tree1  Tree2  Tree3  Tree4  Tree5
        │      │      │      │      │
        ▼      ▼      ▼      ▼      ▼
      $20k   $22k   $19k   $21k   $20k
        │      │      │      │      │
        └──────┴──────┼──────┴──────┘
                      ▼
               Average: $20.4k
```

---

## How to Run the Project

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Train the Model (only needed once)
```bash
# Download dataset first (requires Kaggle API key)
kaggle datasets download deepcontractor/car-price-prediction-challenge -p data --unzip

# Run training
python train_model.py
```

### Step 2: Start the API
```bash
uvicorn main:app --reload
```

### Step 3: Test the API

**Option A**: Open browser at http://127.0.0.1:8000/docs

**Option B**: Use PowerShell
```powershell
# Health check
Invoke-RestMethod -Uri "http://127.0.0.1:8000/"

# Make prediction
$body = @{
    Levy=500
    Manufacturer="TOYOTA"
    Model="Corolla"
    Prod_year=2015
    Category="Sedan"
    Leather_interior="Yes"
    Fuel_type="Petrol"
    Engine_volume=1.8
    Mileage=50000
    Cylinders=4
    Gear_box_type="Automatic"
    Drive_wheels="Front"
    Doors=4
    Wheel="Left wheel"
    Color="Black"
    Airbags=6
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/" -Method Post -Body $body -ContentType "application/json"
```

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check, API status |
| GET | `/health` | Detailed health check |
| GET | `/features` | List of expected features |
| POST | `/predict/` | Make a price prediction |
| GET | `/docs` | Swagger UI (interactive docs) |
| GET | `/redoc` | ReDoc UI (alternative docs) |

### Example Request/Response

**Request:**
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

---

## Files Changed

### New Files Created

| File | Purpose |
|------|---------|
| `train_model.py` | Complete training script with data cleaning |
| `data/` | Folder for storing dataset |
| `PROJECT_GUIDE.md` | This learning documentation |

### Files Modified

| File | Changes Made |
|------|--------------|
| `main.py` | Complete rewrite with proper documentation, error handling, new endpoints |
| `car_price_model.pkl` | Retrained model (1,518 features instead of 6,737) |
| `preprocessor.pkl` | New ColumnTransformer instead of StandardScaler |

### Files Unchanged

| File | Reason |
|------|--------|
| `requirements.txt` | Dependencies were already correct |
| `.gitignore` | Already properly configured |

---

## What I Learned

### 1. Data Preprocessing is Critical
- The model is only as good as the data preparation
- Mismatched preprocessing = broken predictions

### 2. Feature Engineering Matters
- Numeric features need scaling
- Categorical features need encoding
- Column names must be consistent

### 3. Always Test End-to-End
- Unit testing the model isn't enough
- Must test the full pipeline: API → Preprocessing → Model → Response

### 4. Documentation Saves Time
- Clear comments explain the "why", not just the "what"
- Pydantic models generate API docs automatically

### 5. Version Compatibility
- Pickle files are version-sensitive
- Save the scikit-learn version used for training

---

## Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | $5,113.71 | Average error of ~$5k |
| RMSE | $10,101.00 | Larger errors penalized more |
| R² Score | 0.6726 | Explains 67% of price variance |

### Top 5 Most Important Features

1. **Production Year** (42.5%) - Newer cars cost more
2. **Airbags** (19.4%) - Safety features add value
3. **Engine Volume** (4.1%) - Larger engines = higher price
4. **Model** (varies) - Some models are premium
5. **Mileage** (1.8%) - Lower mileage = higher price

---

## Future Improvements

1. **Better Model**: Try XGBoost or LightGBM for better accuracy
2. **Feature Engineering**: Create new features like "car age"
3. **API Features**: Add batch prediction endpoint
4. **Deployment**: Deploy to Render or AWS
5. **Monitoring**: Add logging and performance tracking

---

*Last Updated: February 2026*
