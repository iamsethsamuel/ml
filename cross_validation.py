import pandas as pd
import kagglehub
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

print("Downloading dataset")
path = kagglehub.dataset_download("dansbecker/melbourne-housing-snapshot")

print("Read and converting dataset to CSV")
melbourne_data = pd.read_csv(path+"/melb_data.csv")

y = melbourne_data.Price
X = melbourne_data.drop(["Price"], axis=1)

print("Splitting test and validation data")
# split data into training and validation data
full_train_x, full_val_x, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)
cols_with_missing = [col for col in full_train_x.columns if full_train_x[col].isnull().any()]
full_train_x.drop(cols_with_missing, axis=1, inplace=True)

categorical_cols = [col for col in full_train_x.columns if full_train_x[col].dtype == "object"]
numerical_cols = [col for col in full_train_x.columns if full_train_x[col].dtype in ["int64", "float64"]]

cols = categorical_cols + numerical_cols
train_x = full_train_x[cols].copy()
val_x = full_val_x[cols].copy()


# transforms text data to numerical values
imputer = SimpleImputer(strategy="constant")

categorical_transformer = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("nums", imputer, numerical_cols),
        ("chars", categorical_transformer, cols)
    ]
)

mp = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, random_state=0))
])

scores = -1 * cross_val_score(mp, X, y, cv=5, scoring="neg_mean_absolute_error")

print(f"MAE scores: {scores}")