"""
Car Price Prediction - Model Training Script
=============================================
This script trains a machine learning model to predict car prices.

Learning Objectives:
1. Data Loading and Exploration
2. Data Cleaning and Preprocessing
3. Feature Engineering with ColumnTransformer
4. Model Training with RandomForestRegressor
5. Model Evaluation and Saving

Author: Shaik Tanzeel Ahmed
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load the Dataset
# ============================================================================
print("=" * 60)
print("STEP 1: Loading the Dataset")
print("=" * 60)

# Load the training data
# The dataset contains car listings with features like manufacturer, model, year, etc.
df = pd.read_csv('data/car_price_prediction.csv')

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================================
# STEP 2: Data Exploration
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Data Exploration")
print("=" * 60)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics for Numeric Columns:")
print(df.describe())

# ============================================================================
# STEP 3: Data Cleaning
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Data Cleaning")
print("=" * 60)

# Make a copy to avoid modifying original data
df_clean = df.copy()

# 3.1 Handle the 'Levy' column (has '-' for missing values)
print("\n3.1 Cleaning 'Levy' column...")
print(f"   Unique Levy values sample: {df_clean['Levy'].unique()[:10]}")
df_clean['Levy'] = pd.to_numeric(df_clean['Levy'], errors='coerce')
df_clean['Levy'] = df_clean['Levy'].fillna(0)  # Fill missing with 0
print(f"   After cleaning - Levy range: {df_clean['Levy'].min()} to {df_clean['Levy'].max()}")

# 3.2 Clean 'Mileage' column (remove ' km' suffix and convert to numeric)
print("\n3.2 Cleaning 'Mileage' column...")
print(f"   Sample before: {df_clean['Mileage'].head(3).tolist()}")
df_clean['Mileage'] = df_clean['Mileage'].str.replace(' km', '', regex=False)
df_clean['Mileage'] = pd.to_numeric(df_clean['Mileage'], errors='coerce')
df_clean['Mileage'] = df_clean['Mileage'].fillna(df_clean['Mileage'].median())
print(f"   Sample after: {df_clean['Mileage'].head(3).tolist()}")

# 3.3 Clean 'Engine volume' column (extract numeric value, handle 'Turbo')
print("\n3.3 Cleaning 'Engine volume' column...")
print(f"   Sample before: {df_clean['Engine volume'].unique()[:10]}")
# Extract the numeric part (e.g., "2.0 Turbo" -> 2.0)
df_clean['Engine volume'] = df_clean['Engine volume'].str.extract(r'(\d+\.?\d*)')[0]
df_clean['Engine volume'] = pd.to_numeric(df_clean['Engine volume'], errors='coerce')
df_clean['Engine volume'] = df_clean['Engine volume'].fillna(df_clean['Engine volume'].median())
print(f"   Sample after: {df_clean['Engine volume'].head(5).tolist()}")

# 3.4 Clean 'Doors' column (handle values like '02-Mar', '04-May')
print("\n3.4 Cleaning 'Doors' column...")
print(f"   Unique values before: {df_clean['Doors'].unique()}")
# Map door values to numeric
door_mapping = {
    '02-Mar': 2, '04-May': 4, '>5': 5, '2': 2, '3': 3, '4': 4, '5': 5,
    2: 2, 3: 3, 4: 4, 5: 5
}
df_clean['Doors'] = df_clean['Doors'].map(door_mapping)
df_clean['Doors'] = df_clean['Doors'].fillna(4)  # Default to 4 doors
print(f"   Unique values after: {df_clean['Doors'].unique()}")

# 3.5 Rename columns to match API naming convention
print("\n3.5 Renaming columns for API consistency...")
column_mapping = {
    'Prod. year': 'Prod_year',
    'Leather interior': 'Leather_interior',
    'Fuel type': 'Fuel_type',
    'Engine volume': 'Engine_volume',
    'Gear box type': 'Gear_box_type',
    'Drive wheels': 'Drive_wheels'
}
df_clean = df_clean.rename(columns=column_mapping)
print(f"   New column names: {df_clean.columns.tolist()}")

# 3.6 Remove unnecessary columns
print("\n3.6 Removing ID column (not needed for prediction)...")
df_clean = df_clean.drop(columns=['ID'])

# ============================================================================
# STEP 4: Define Features and Target
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Defining Features and Target")
print("=" * 60)

# Target variable (what we want to predict)
target = 'Price'

# Feature columns (what we use to make predictions)
feature_columns = [col for col in df_clean.columns if col != target]

# Separate features (X) and target (y)
X = df_clean[feature_columns]
y = df_clean[target]

print(f"Target variable: {target}")
print(f"Number of features: {len(feature_columns)}")
print(f"Feature columns: {feature_columns}")

# ============================================================================
# STEP 5: Split Data into Training and Testing Sets
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Splitting Data")
print("=" * 60)

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# ============================================================================
# STEP 6: Create Preprocessing Pipeline
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Creating Preprocessing Pipeline")
print("=" * 60)

# Define which columns are numeric and which are categorical
numeric_features = ['Levy', 'Prod_year', 'Engine_volume', 'Mileage', 'Cylinders', 'Doors', 'Airbags']
categorical_features = ['Manufacturer', 'Model', 'Category', 'Leather_interior', 
                        'Fuel_type', 'Gear_box_type', 'Drive_wheels', 'Wheel', 'Color']

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")

# Create the ColumnTransformer
# - StandardScaler: Normalizes numeric features (mean=0, std=1)
# - OneHotEncoder: Converts categorical features to binary columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'  # Drop any columns not specified
)

print("\nPreprocessor created successfully!")
print("- Numeric features will be scaled using StandardScaler")
print("- Categorical features will be encoded using OneHotEncoder")

# ============================================================================
# STEP 7: Fit Preprocessor and Transform Data
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: Fitting Preprocessor")
print("=" * 60)

# Fit the preprocessor on training data and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Features after preprocessing: {X_train_processed.shape[1]}")
print(f"  - Numeric features: {len(numeric_features)}")
print(f"  - One-hot encoded features: {X_train_processed.shape[1] - len(numeric_features)}")

# ============================================================================
# STEP 8: Train the Model
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: Training the Model")
print("=" * 60)

# Create and train RandomForestRegressor
# Parameters explained:
# - n_estimators: Number of trees in the forest (more trees = better accuracy but slower)
# - max_depth: Maximum depth of each tree (prevents overfitting)
# - min_samples_split: Minimum samples required to split a node
# - random_state: Ensures reproducibility
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # Use all CPU cores for faster training
)

print("Training RandomForestRegressor...")
print("Parameters:")
print(f"  - n_estimators: 100")
print(f"  - max_depth: 15")
print(f"  - min_samples_split: 5")

model.fit(X_train_processed, y_train)
print("\nModel training complete!")

# ============================================================================
# STEP 9: Evaluate the Model
# ============================================================================
print("\n" + "=" * 60)
print("STEP 9: Evaluating the Model")
print("=" * 60)

# Make predictions on test set
y_pred = model.predict(X_test_processed)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"  - Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"  - Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"  - R² Score: {r2:.4f} ({r2*100:.2f}%)")

print("\nInterpretation:")
print(f"  - On average, predictions are off by ${mae:,.2f}")
print(f"  - The model explains {r2*100:.1f}% of the variance in car prices")

# ============================================================================
# STEP 10: Feature Importance
# ============================================================================
print("\n" + "=" * 60)
print("STEP 10: Feature Importance Analysis")
print("=" * 60)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get feature importances from the model
importances = model.feature_importances_

# Create DataFrame and sort by importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# STEP 11: Save the Model and Preprocessor
# ============================================================================
print("\n" + "=" * 60)
print("STEP 11: Saving Model and Preprocessor")
print("=" * 60)

# Save the trained model
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to 'car_price_model.pkl'")

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Preprocessor saved to 'preprocessor.pkl'")

# ============================================================================
# STEP 12: Test the Saved Model
# ============================================================================
print("\n" + "=" * 60)
print("STEP 12: Testing Saved Model")
print("=" * 60)

# Create a sample input (like what the API would receive)
sample_input = pd.DataFrame([{
    'Levy': 500,
    'Manufacturer': 'TOYOTA',
    'Model': 'Corolla',
    'Prod_year': 2015,
    'Category': 'Sedan',
    'Leather_interior': 'Yes',
    'Fuel_type': 'Petrol',
    'Engine_volume': 1.8,
    'Mileage': 50000,
    'Cylinders': 4,
    'Gear_box_type': 'Automatic',
    'Drive_wheels': 'Front',
    'Doors': 4,
    'Wheel': 'Left wheel',
    'Color': 'Black',
    'Airbags': 6
}])

print("Sample input:")
print(sample_input.to_string())

# Transform and predict
sample_processed = preprocessor.transform(sample_input)
predicted_price = model.predict(sample_processed)[0]

print(f"\nPredicted Price: ${predicted_price:,.2f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nFiles created:")
print("  1. car_price_model.pkl - The trained RandomForest model")
print("  2. preprocessor.pkl - The ColumnTransformer for data preprocessing")
print("\nYou can now run the FastAPI server with: uvicorn main:app --reload")
