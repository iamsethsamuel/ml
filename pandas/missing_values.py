import pandas as pd;
import kagglehub
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

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
predicators = melbourne_data.drop(["Price"], axis=1)
X = predicators.select_dtypes(exclude="object")

print("Splitting test and validation data")
# split data into training and validation data
train_x, val_x, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)



# Drop Columns with Missing Values


# drop the columns with missing values
cols_with_missing_values = [cols for cols in train_x.columns if melbourne_data[cols].isnull().any() ]
print("Columns with missing values: ", cols_with_missing_values)

reduced_X_train = train_x.drop(cols_with_missing_values, axis=1, errors='ignore')
reduced_X_val = val_x.drop(cols_with_missing_values, axis=1)



# remove 'CouncilArea' from the list
    
print("\r\nMAE from Approach 1 (Drop columns with missing values): ")
print(score_dataset(reduced_X_train, reduced_X_val, train_y, val_y))
    

# Imputation
imputer = SimpleImputer()
imputed_x_train = pd.DataFrame(imputer.fit_transform(train_x))
imputed_x_val = pd.DataFrame(imputer.fit_transform(val_x))

# Imputation removed column names; put them back
imputed_x_train.columns = train_x.columns
imputed_x_val.columns = val_x.columns

print("\n\r MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_x_train, imputed_x_val, train_y, val_y))


# An Extension to Imputation

extended_x_train = train_x.copy()
extended_x_val = val_x.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing_values:
    extended_x_train[col + "_was_missing"] = extended_x_train[col].isnull()
    extended_x_val[col + "_was_missing"] = extended_x_val[col].isnull()
    
extented_imputed_x_train = pd.DataFrame(imputer.fit_transform(extended_x_train))
extended_imputed_x_val = pd.DataFrame(imputer.fit_transform(extended_x_val))

extented_imputed_x_train.columns = extended_x_train.columns
extended_imputed_x_val.columns = extended_x_val.columns
print("\n\r MAE from Approach 3 (Imputation with extended columns):")
print(score_dataset(extented_imputed_x_train, extended_imputed_x_val, train_y, val_y))