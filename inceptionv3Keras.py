import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from constants import IMAGE_WIDTH,IMAGE_HEIGHT,NUM_KEYS

def InceptionV3(image_height=IMAGE_HEIGHT,image_width=IMAGE_WIDTH,load_weights=False):
    if load_weights:

        base_model=tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            pooling=None
        )
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(NUM_KEYS, activation='softmax')(x)

        for layer in base_model.layers:
            layer.trainable = False
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)


    else:
        model=tf.keras.applications.InceptionV3(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            pooling=None,
            classes=9,
            classifier_activation="softmax",
        )

    return model