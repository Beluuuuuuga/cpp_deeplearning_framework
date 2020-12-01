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

if __name__ == "__main__":

    model = load_model("models/base_model.hdf5")
    # Numpy load
    loaded_sample = np.load('data/sample.npy')
    
    print(loaded_sample.shape)
    print(model.predict(loaded_sample))

    output = get_output(model, 'conv2', loaded_sample)
    convert_conv2_output = convert_dense_output(output).ravel()
    # print(convert_conv2_output[:20])
    for i in range(40):
        print(convert_conv2_output[i])
    
