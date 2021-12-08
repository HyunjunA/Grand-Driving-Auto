import os, sys
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import numpy as np

input_dir = sys.argv[1]
output_dir = sys.argv[2]
# output_dir = '/content/drive/MyDrive/GDA/Feature/'

# TFHub feature vector model
url = 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5'
base_model = hub.KerasLayer(url, input_shape=(300, 400, 3))
base_model.trainable = False

model = keras.Sequential([base_model])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# feature extraction function
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers],
)


# save the features to output files
def save_file(output_file, data):
    with open(output_file, 'wb'):
        np.save(output_file, data)


def start_feature_extraction_from_training_data():

    for file_num in range(1, 403):
        if file_num == 7:
            continue
        file_path = os.path.join(input_dir, f"training_data-{file_num}.npy")
        file_data = np.load(file_path, allow_pickle=True)



        start_time = time.time()
        #input_images = np.array(list(file_data[:, 0]), dtype=np.float)
        #label = np.array(list(file_data[:, 1]), dtype=np.int)
        #reshaped_image = np.resize(input_image, (500, 299, 299, 3))
        output_data = []
        output_features = []
        for input_image, input_label in file_data:

            input_image = input_image.reshape(1, 300, 400, 3)
            extracted_output = feature_extractor(input_image)  # Call feature extractor function here
            output_features.append([extracted_output, input_label])

        output_data.append(output_features)

        end_time = time.time()
        print("File No. ", file_num, " | Processing time: ", end_time - start_time)
        export_file = os.path.join(output_dir, f"extracted_data-{file_num}.npy")
        #print(output_data)
        save_file(export_file, output_data)


start_feature_extraction_from_training_data()