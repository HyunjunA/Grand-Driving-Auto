import tensorflow as tf
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout,Activation
from tensorflow.keras import activations

def alexnetv2(image_height=IMAGE_HEIGHT,image_width=IMAGE_WIDTH):
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        Conv2D(filters=96, kernel_size=11, strides=4,input_shape=(image_height, image_width , 3)),
        Activation('relu'),
        MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        Conv2D(filters=256, kernel_size=5, padding='same'),
        Activation('relu'),
        MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        Conv2D(filters=384, kernel_size=3, padding='same'),
        Activation('relu'),
        Conv2D(filters=384, kernel_size=3, padding='same'),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding='same'),
        Activation('relu'),
        MaxPool2D(pool_size=3, strides=2),
        BatchNormalization(),
        Flatten(),
        Dense(4096),
        Activation('relu'),
        Dropout(0.5),
        Dense(4096),
        Activation('relu'),
        Dropout(0.5),
        Dense(4096),
        Activation('relu'),
        Dropout(0.5),
        # Output layer. 9 classes for 9 different keypresses
        Dense(9,activation='softmax')])