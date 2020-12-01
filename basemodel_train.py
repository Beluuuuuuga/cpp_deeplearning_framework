import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Convolution2D, Activation


def base_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1), name='conv1')) # test1
    model.add(layers.MaxPooling2D((2, 2), name='maxpool1')) # test2
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same", name='conv2')) # test3
    model.add(layers.MaxPooling2D((2, 2), name='maxpool2')) # test4
    model.add(Flatten(name='flatten1')) 

    model.add(Dense(1024, activation='relu', name='dense1')) # test5
    model.add(layers.Dense(10, activation='softmax', name='dense2')) # test6
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # ピクセルの値を 0~1 の間に正規化
    train_images_regularized, test_images_regularized = train_images / 255.0, test_images / 255.0

    model = base_model()
    model.fit(train_images_regularized, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
    print(test_acc)

    model.save("models/base_model.hdf5")

    loaded_model = load_model("models/base_model.hdf5")
