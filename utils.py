from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def score_dataset(x_train, x_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    print("Training model")
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    return mean_absolute_error(y_val, pred)
