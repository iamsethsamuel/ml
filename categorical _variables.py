import pandas as pd;
import kagglehub
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


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
train_x, val_x, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)

# Get list of categorical variables
cat_var = (train_x.dtypes == 'object')
object_cols = list(cat_var[cat_var].index)
print("Cat_Var: ", cat_var)
print("Categorical variables:", object_cols)

# def score_dataset(x_train, x_val, y_train, y_val):
#     model = RandomForestRegressor(n_estimators=10, random_state=0)
#     print("Training model")
#     model.fit(x_train, y_train)
#     pred = model.predict(x_val)
#     return mean_absolute_error(y_val, pred)




# Imputation
# imputer = SimpleImputer()
# imputed_x_train = pd.DataFrame(imputer.fit_transform(train_x))
# imputed_x_val = pd.DataFrame(imputer.fit_transform(val_x))

# # Imputation removed column names; put them back
# imputed_x_train.columns = train_x.columns
# imputed_x_val.columns = val_x.columns

# print("\n\r MAE from Approach 2 (Imputation):")
# print(score_dataset(imputed_x_train, imputed_x_val, train_y, val_y))
