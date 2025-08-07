import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

# Load the trained model
model = keras.models.load_model("./models/hotel_model.keras")

# Load the original dataset
original_data = pd.read_csv('./datasets/hotel.csv')

# Take a random sample of 10 rows
sample_df = original_data.sample(n=100) # Use a fixed random_state for reproducibility
print("Sampled DataFrame (10 rows):")
print(sample_df)

# Separate features (X) and the actual target (y)
X_sample = sample_df.copy()
y_actual = X_sample.pop('is_canceled')

# Define feature columns
features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
]
features_cat = [
    "hotel", "arrival_date_month", "meal",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]

# Preprocessing from the original script
# Map month names to numbers
X_sample['arrival_date_month'] = \
    X_sample['arrival_date_month'].map(
        {'January':1, 'February': 2, 'March':3,
         'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10,
         'November':11, 'December':12}
    )

# Create the preprocessing pipelines
transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

# Create the column transformer
preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# Fit the preprocessor on the full original data to ensure consistent encoding
X_original = original_data.copy()
X_original.pop('is_canceled')
X_original['arrival_date_month'] = \
    X_original['arrival_date_month'].map(
        {'January':1, 'February': 2, 'March':3,
         'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10,
         'November':11, 'December':12}
    )
preprocessor.fit(X_original)

# Transform the sample data
X_processed = preprocessor.transform(X_sample)

# Make predictions
predictions_proba = model.predict(X_processed)
predictions = (predictions_proba > 0.5).astype(int) # Convert probabilities to 0 or 1

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    'Actual': y_actual.values,
    'Predicted': predictions.flatten()
})

print("\nComparison of Actual vs. Predicted:")
print(comparison_df)

def calculate_accuracy(df):
    """Calculates the prediction accuracy from a comparison DataFrame."""
    correct_predictions = (df['Actual'] == df['Predicted']).sum()
    total_predictions = len(df)
    if total_predictions == 0:
        return 0.0
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

accuracy_percentage = calculate_accuracy(comparison_df)
print(f"\nPrediction Accuracy: {accuracy_percentage:.2f}%")
