import pandas as pd;
import kagglehub
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("dansbecker/melbourne-housing-snapshot")


melbourne_data = pd.read_csv(path+"/melb_data.csv")

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# column to use for predicting the data
y = melbourne_data.Price

# colums that the model will use to predict the price
# these are the features
# we will use the following columns to predict the price
melbourne_features = [ "Rooms", "Bathroom",  "BuildingArea", "YearBuilt", "Car", "Postcode", 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]


# split data into training and validation data
train_x, val_x, train_y, val_y = train_test_split(X, y, random_state=0)

print("Initializing the model")
# DecisionTreeRegressor is a model that uses a decision tree to make predictions
# random_state is a seed for the random number generator
# so that the results are reproducible
# you can use any number you want
# but using 1 is a common practice
model = DecisionTreeRegressor(random_state=1)

print("Training the model...")
# fit() method is used to train the model
model.fit(train_x, train_y)


predictions = model.predict(val_x)
print("The mean absolute error of the model is:")
print(mean_absolute_error(val_y, predictions))