from email.mime import image
import kagglehub
import os, warnings
import keras
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import pandas as pd

print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("ryanholbrook/car-or-truck")
print("Downloading model...")
model_path = kagglehub.dataset_download("ryanholbrook/cv-course-models")
print((model_path))

ds_train_ = image_dataset_from_directory(
    dataset_path+"/train",
    labels="inferred",
    label_mode="binary",
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
)

ds_valid_ = image_dataset_from_directory(
    dataset_path+"/valid",
    labels="inferred",
    label_mode="binary",
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
)


# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

print("Converting training images to float32...")
ds_train = (
    ds_train_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
)

print("Converting validation images to float32...")
ds_valid = (
    ds_valid_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
)

print("Creating base model...")

base_model = tf.keras.layers.TFSMLayer(
    model_path + "/cv-course-models/inceptionv1",
    call_endpoint="serving_default"
)

base_model.trainable = False

print("Creating the network...")
model = keras.Sequential([
    base_model,
    layers.Lambda(lambda x: x['keras_layer_1']),
    layers.Flatten(),
    layers.Dense(6, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

optimizer = tf.keras.optimizers.Adam(epsilon=0.01)

print("Compiling model...")
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)

print("Fitting model...")
fitting = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()