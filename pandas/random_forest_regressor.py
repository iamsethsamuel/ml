import pandas as pd;
import kagglehub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Download latest version
print("Downloading dataset")
path = kagglehub.dataset_download("dansbecker/melbourne-housing-snapshot")

print("Read and converting dataset to CSV")
melbourne_data = pd.read_csv(path+"/melb_data.csv")

print("Filtering dataset")
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# column to use for predicting the data
y = melbourne_data.Price

# colums that the model will use to predict the price
# these are the features
# we will use the following columns to predict the price
melbourne_features = [ "Rooms", "Bathroom",  "BuildingArea", "YearBuilt", "Car", "Postcode", 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

print("Splitting test and validation data")
# split data into training and validation data
train_x, val_x, train_y, val_y = train_test_split(X, y, random_state=0)

print("Initializing model...")
model = RandomForestRegressor(random_state=1)

print("Training...")
model.fit(train_x, train_y)

predictions = model.predict(val_x)
print("Mean Absolute Error", mean_absolute_error(val_y, predictions))