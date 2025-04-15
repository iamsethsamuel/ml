import pandas as pd;
import kagglehub
from sklearn.tree import DecisionTreeRegressor
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
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    """
    Function to get the mean absolute error of the model
    :param max_leaf_nodes: maximum number of leaf nodes
    :param train_x: training data
    :param val_x: validation data
    :param train_y: training labels
    :param val_y: validation labels
    :return: mean absolute error of the model
    """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    predictions = model.predict(val_x)
    mae = mean_absolute_error(val_y, predictions)
    return mae


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
curr_mae = (candidate_max_leaf_nodes[0], 0)
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_node in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf_node, train_X, val_X, train_y, val_y)
    if(curr_mae[1]==0):
        curr_mae =(max_leaf_node,mae)
    elif(mae < curr_mae[1]):
        curr_mae = (max_leaf_node, mae)
        
        
    

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = curr_mae[0]
print ("Best Tree Size: ", curr_mae)

# Create the final model with the best value of max_leaf_nodes
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size)

# and fit it to the whole dataset
final_model.fit(X, y)
