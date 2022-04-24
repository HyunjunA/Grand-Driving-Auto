import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from constants import IMAGE_WIDTH,IMAGE_HEIGHT

 
def svm(image_height=IMAGE_HEIGHT,image_width=IMAGE_WIDTH):

    return tf.keras.Sequential(
    [
        tf.keras.Input(shape=(image_height,image_width)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=10),
    ]
)

