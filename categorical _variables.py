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


print("Splitting test and validation data")
# split data into training and validation data
full_train_x, full_val_x, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)

cols_with_missing = [col for col in full_train_x.columns if full_train_x[col].isnull().any()]
full_train_x.drop(cols_with_missing, axis=1, inplace=True)
full_val_x.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cols for cols in full_train_x.columns if full_train_x[cols].nunique() < 10 and full_train_x[cols].dtype == "object"]

# Select numerical columns
numerical_cols = [col for col in full_train_x.columns if full_train_x[col].dtype in ["int64", "float64"]]

cols = low_cardinality_cols + numerical_cols
train_x = full_train_x[cols].copy()
val_x = full_val_x[cols].copy()




# Ordinal Encoding
l_train_x = train_x.copy()
l_val_x = val_x.copy()

# Get list of categorical variables (Columns with strings as rows)
cat_var = (l_train_x.dtypes == 'object')
object_cols = list(cat_var[cat_var].index)

ordinal = OrdinalEncoder()
l_train_x[object_cols] = ordinal.fit_transform(train_x[object_cols])
l_val_x[object_cols] = ordinal.transform(val_x[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(l_train_x,l_val_x, train_y, val_y))



# Hot Encoding

hot_endoder =  OneHotEncoder(handle_unknown='ignore', sparse_output=False)
hot_train = pd.DataFrame(hot_endoder.fit_transform(l_train_x[object_cols]))
hot_val = pd.DataFrame(hot_endoder.transform((l_val_x[object_cols])))

# One-hot encoding removed index; put it back
hot_train.index = train_x.index
hot_val.index = val_x.index


# Remove categorical columns (will replace with one-hot encoding)
num_x_train = train_x.drop(object_cols, axis=1)
num_val_x = val_x.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
hot_x_train = pd.concat([hot_train, num_x_train], axis=1)
hot_x_val = pd.concat([hot_val, num_val_x], axis=1)

# Ensure all columns have string type
hot_x_train.columns = hot_x_train.columns.astype(str)
hot_x_val.columns = hot_x_val.columns.astype(str)


print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(hot_x_train, hot_x_val,train_y, val_y))