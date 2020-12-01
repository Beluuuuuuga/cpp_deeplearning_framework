
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Convolution2D, Activation

if __name__ == "__main__":

    model = load_model("models/base_model.hdf5")

    # 重み
    w0 = np.array(model.get_weights()[0]).ravel()
    w1 = np.array(model.get_weights()[2]).ravel()
    w2 = np.array(model.get_weights()[4]).ravel()
    w3 = np.array(model.get_weights()[6]).ravel()
    # バイアス
    b0 = np.array(model.get_weights()[1]).ravel()
    b1 = np.array(model.get_weights()[3]).ravel()
    b2 = np.array(model.get_weights()[5]).ravel()
    b3 = np.array(model.get_weights()[7]).ravel()