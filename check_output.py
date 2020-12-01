import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Convolution2D, Activation

def get_output(model, layer_name, sample):
    model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    return model.predict(sample)

def convert_dense_output(output):
    return np.transpose(output, axes=[3,1,2,0])

def convert_conv(w):
    w = np.transpose(w, axes=[3,2,0,1])
    return w


if __name__ == "__main__":

    model = load_model("models/base_model.hdf5")
    # Numpy load
    loaded_sample = np.load('data/sample.npy')
    
    print("sample image shape:",loaded_sample.shape)
    print(model.predict(loaded_sample))

    # 出力
    conv2_output = get_output(model, 'conv2', loaded_sample)
    convert_conv2_output = convert_dense_output(conv2_output).ravel()

    pool1_output = get_output(model, 'maxpool1', loaded_sample)
    print("maxpool1 output shape:",pool1_output.shape)

    flatten_output = get_output(model, 'flatten1', loaded_sample)
    print("flatten output shape:",flatten_output.shape)

    dense1_output = get_output(model, 'dense1', loaded_sample)
    print("dense1 output shape:",dense1_output.shape)

    # 重み
    dense1_weight = np.array(model.get_weights()[4])
    print("dense1 weight shape:",dense1_weight.shape)




    # print(convert_conv2_output[:20])
    # for i in range(40):
    #     print(convert_conv2_output[i])
    
