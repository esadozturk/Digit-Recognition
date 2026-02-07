import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()