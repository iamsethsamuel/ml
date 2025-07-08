from tabnanny import verbose
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd;
import kagglehub
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utils import score_dataset

# Download latest version
print("Downloading dataset")
path = kagglehub.dataset_download("dansbecker/melbourne-housing-snapshot")

print("Read and converting dataset to CSV")
melbourne_data = pd.read_csv(path+"/melb_data.csv")



print("Filtering dataset")

# column to use for predicting the data
y = melbourne_data.Price

# we'll use only numerical predictors
X = melbourne_data.drop(["Price"], axis=1)

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = melbourne_data[cols_to_use]

# Select target
y = melbourne_data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

print("Building the model")

model = XGBRegressor(n_estimators=500, learning_rate=0.03, n_jobs=3, enable_categorical=True, )

print("Fitting the model...")
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

actual_price = melbourne_data.iloc[[0]].Price

print(f"Actual price of the first house: ${actual_price.iloc[0]:,.2f}")

mp = model.predict(X.iloc[[0]])
print(f"The predicted price is: ${mp[0]:,.2f}")


print("Predicting...")
preds = model.predict(X_valid)



print("MAE Score")
print(mean_absolute_error(y_valid, preds))