import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def make_benchmark_sample():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # ピクセルの値を 0~1 の間に正規化
    train_images_regularized, test_images_regularized = train_images / 255.0, test_images / 255.0

    sample = train_images_regularized[0]

    # PNG形式
    sample_png = sample.reshape((28, 28))
    cv2.imwrite('data/sample.png',sample_png)

    # TXT形式
    sample_txt = sample.ravel()
    np.savetxt('data/tmp.txt',sample_txt, fmt='%f4')

    # Numpy形式
    sample_np = sample.reshape((1, 28, 28, 1))
    np.save('data/sample', sample_np)

    # Numpy load
    loaded_sample = np.load('data/sample.npy')
    print(loaded_sample.shape)

if __name__ == "__main__":
    make_benchmark_sample()