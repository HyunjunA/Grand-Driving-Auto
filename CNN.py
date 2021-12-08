import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
    
def cnn(image_height=IMAGE_HEIGHT,image_width=IMAGE_WIDTH):

    return tf.keras.Sequential([
        layers.Conv2D(96, (11, 11),strides=4, activation='relu', input_shape=(image_height, image_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(9,activation=tf.keras.activations.softmax)
    ])
