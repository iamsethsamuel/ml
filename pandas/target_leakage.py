import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

print("Downloading dataset")
path = kagglehub.dataset_download("rainsun/aer-credit-card-datacsv")

print("Read and converting dataset to CSV")
credit_data = pd.read_csv(path+"/AER_credit_card_data.csv", true_values=["yes"], false_values=["no"])

potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
    
y = credit_data.card
X = credit_data.drop(["card"], axis=1)
X_without_leaks = X.drop(potential_leaks, axis=1)

# use cross validation to improve model quality due to small dataset
mp = make_pipeline(RandomForestClassifier(n_estimators=100))
scores = cross_val_score(mp, X, y, cv=5, scoring="accuracy")

print("Scores with Data leak...")
print(scores.mean())

expenditure_cols = X.expenditure[y]
no_expenditure_cols = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditure_cols == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( no_expenditure_cols == 0).mean()))

print("Without leaks")

scores = cross_val_score(mp, X_without_leaks, y, cv=5, scoring="accuracy")

print("Scores without Data Leak...")
print(scores.mean())
