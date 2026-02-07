import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape(60000, 28*28)
train_data = train_data.astype("float32") / 255
train_labels = train_labels.reshape(60000, 28*28)
train_labels = train_labels.astype("float32") / 255

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_data, train_labels, epochs=7, batch_size=128)