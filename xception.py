import os
import sys
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from constants import IMAGE_WIDTH,IMAGE_HEIGHT

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr


# def get_model(session):
def xception(image_height=IMAGE_HEIGHT,image_width=IMAGE_WIDTH):
    # create the base pre-trained model
    base_model = Xception(weights=None, include_top=False, input_shape=(image_height, image_width, 3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # putput layer
    # predictions = Dense(session.training_dataset_info['number_of_labels'], activation='softmax')(x)
    predictions = Dense(9, activation='softmax')(x)
    # model
    model = Model(inputs=base_model.input, outputs=predictions)

    # learning_rate = 0.001
    # opt = keras.optimizers.adam(lr=learning_rate, decay=1e-5)

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=opt,
    #               metrics=['accuracy'])

    return model