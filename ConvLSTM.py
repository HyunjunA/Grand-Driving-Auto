from constants import IMAGE_WIDTH,IMAGE_HEIGHT,NUM_KEYS
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D, MaxPooling2D, TimeDistributed, Dense, Flatten,Dropout

def convLSTM(seq_len):

    model = Sequential()
    model.add(Input(shape=(seq_len, IMAGE_HEIGHT, IMAGE_WIDTH,3)))
    model.add(ConvLSTM2D(filters = 20, kernel_size = (3, 3), padding='same',
                         return_sequences= True, data_format= "channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

    model.add(ConvLSTM2D(filters=10, kernel_size=(3, 3)
                                 , data_format='channels_last'
                                 , padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1,2, 2), padding='same', data_format='channels_last'))
    model.add(ConvLSTM2D(filters=10, kernel_size=(3, 3)
                         , data_format='channels_last'
                         , padding='same', return_sequences=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(NUM_KEYS, activation= "softmax"))

    return model
