
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Convolution2D, Activation

def convert_conv(w):
    w = np.transpose(w, axes=[3,2,0,1])
    return w


if __name__ == "__main__":

    model = load_model("models/base_model.hdf5")

    # 重み
    w0 = np.array(model.get_weights()[0])
    w1 = np.array(model.get_weights()[2])
    w2 = np.array(model.get_weights()[4])
    w3 = np.array(model.get_weights()[6])

    print("conv1 shape:",w0.shape)
    print("conv2 shape:",w1.shape)
    print("dense1 shape:",w2.shape)
    print("dense2 shape:",w3.shape)

    # TF形式=>My推論モデル
    transposed_w0 = convert_conv(w0)  # 16, 1, 3, 3
    transposed_w1 = convert_conv(w1)  # 32, 16, 3, 3
    w2 = np.transpose(w2, axes=[1,0]) # 1024, 1568
    w3 = np.transpose(w3, axes=[1,0]) # 10, 1024

    print("transposed conv1 shape:",transposed_w0.shape)
    print("transposed conv2 shape:",transposed_w1.shape)
    print("transposed dense1 shape:",w2.shape)
    print("transposed dense2 shape:",w3.shape)

    # My推論モデル=>1次元
    transposed_w0 = transposed_w0.ravel()
    transposed_w1 = transposed_w1.ravel()
    w2 = w2.ravel() # 1605632 
    w3 = w3.ravel() # 10240

    # バイアス
    b0 = np.array(model.get_weights()[1]).ravel()
    b1 = np.array(model.get_weights()[3]).ravel()
    b2 = np.array(model.get_weights()[5]).ravel()
    b3 = np.array(model.get_weights()[7]).ravel()

    # txt保存
    # 重み
    np.savetxt('data/w0.txt',transposed_w0, fmt='%f4') # float32
    np.savetxt('data/w1.txt',transposed_w1, fmt='%f4') # float32
    np.savetxt('data/w2.txt',w2, fmt='%f4') # float32
    np.savetxt('data/w3.txt',w3, fmt='%f4') # float32
    # バイアス
    np.savetxt('data/b0.txt',b0, fmt='%f4') # float32
    np.savetxt('data/b1.txt',b1, fmt='%f4') # float32
    np.savetxt('data/b2.txt',b2, fmt='%f4') # float32
    np.savetxt('data/b3.txt',b3, fmt='%f4') # float32

    