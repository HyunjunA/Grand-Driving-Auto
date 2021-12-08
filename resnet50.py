import os
import sys
import tensorflow as tf
from constants import IMAGE_WIDTH,IMAGE_HEIGHT

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr


# def get_model(session):
def ResNet50(image_height=IMAGE_HEIGHT,image_width=IMAGE_WIDTH):
    model=tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(image_height, image_width, 3),
    pooling=None,
    classes=9
    )

    return model