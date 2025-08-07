from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd



print("Read and converting dataset to DataFrame")
data = pd.read_csv("./datasets/red-wine.csv")

rows, columns = data.shape

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[columns - 1]),
])
model.compile(
    optimizer="adam",
    loss="mae",
)
model.fit(data.iloc[:, :-1], data.iloc[:, -1], epochs=10)
print(model.evaluate(data.iloc[:, :-1], data.iloc[:, -1]))

print(model.summary())