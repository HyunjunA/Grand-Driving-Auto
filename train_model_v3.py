import numpy as np
import argparse
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from constants import NUM_SAMPLES_PER_FILE
import os,sys
import os.path
from os import path
from ConvLSTM import convLSTM
from CNN import cnn
from alexnet import alexnet
from alexnetv2 import alexnetv2
from xception import xception
#from inceptionv3 import inception_v3
from inceptionv3Keras import InceptionV3
from resnet50 import ResNet50
from utils import generate_batch_seq,generate_batch
#from tensorflow.keras.applications.inception_v3 import preprocess_input
#from sklearn.utils.class_weight import compute_class_weight
#from sklearn.model_selection import KFold
from time import time
import random
import glob
from constants import IMAGE_HEIGHT,IMAGE_WIDTH

# Train model using static validation dataset



def main():

    physical_devices = tf.config.list_physical_devices('GPU')

    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    """try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass"""


    # Set up arguments
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir', '-d',type=str)
    parser.add_argument('--val_data_dir','-dv')
    parser.add_argument('--num_files_train','-nt',type=int)
    parser.add_argument('--num_files_val','-nv',type=int)
    parser.add_argument('--model_name','-m',type=str,nargs='?',default='AlexNetV2')
    parser.add_argument('--load_sm','-l',action='store_true')
    parser.add_argument('--resize_im','-r',action='store_true')
    parser.add_argument('--epochs','-e',type=int,nargs='?',default=10)
    parser.add_argument('--batch_size','-b',type=int,nargs='?',default=32)
    parser.add_argument('--seq_len','-s',type=int,nargs='?',default=20)
    parser.add_argument('--learning_rate','-lr',type=float,nargs='?',default=0.0001)

    # Training parameters
    args = parser.parse_args()
    if NUM_SAMPLES_PER_FILE % args.seq_len != 0:
        print(f'Sequence length must be factor of number of samples per file ({NUM_SAMPLES_PER_FILE})')
        exit(-1)
    model_name = args.model_name
    load_sm = args.load_sm
    resize = args.resize_im
    data_dir = args.data_dir
    val_data_dir = args.val_data_dir
    num_files_train = args.num_files_train
    num_files_val = args.num_files_val
    epochs = args.epochs
    batch_size = args.batch_size
    seq_len = args.seq_len
    learning_rate = args.learning_rate

    class_weight = {0 : 0.32,
                    1 : 3.63,
                    2 : 1.56,
                    3: 1.78,
                    4: 5.16,
                    5: 3.836,
                    6: 600.0,
                    7: 1.0,
                    8: 0.26
                    }

    #Choose Model
    if resize:
        image_height = 360
        image_width = 480
    else:
        image_height = IMAGE_HEIGHT
        image_width = IMAGE_WIDTH
    if model_name=='ConvLSTM':
        model = convLSTM(seq_len)
    elif model_name== "InceptionV3":
        model = InceptionV3(image_height,image_width)
    elif model_name=="AlexNet":
        model = alexnet(image_height,image_width)
    elif model_name=="AlexNetV2":
        model = alexnetv2(image_height,image_width)
    elif model_name== "Xception":
        model = xception(image_height,image_width)
    elif model_name== "ResNet50":
        model = ResNet50(image_height,image_width)
    else: #CNN model
        model = cnn(image_height,image_width)


    # Load saved model if it exists
    initial_epoch = 0
    if load_sm:
        root_saved_model_local = os.getcwd()+f"/intersavedmodel/{model_name}/"
        if os.path.exists(root_saved_model_local):
            list_of_files = glob.glob(root_saved_model_local+'*')
            latest_file = max(list_of_files, key=os.path.getctime)
            try:
                model = load_model(latest_file)
                print('Loaded saved model file')
                initial_epoch = int(latest_file.split('_')[2])
            except Exception as e:
                print(e)
                print('Could not load saved model')
        else:
            print('Saved model path not found')

    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Set up callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./tb_logs',
        histogram_freq=0,
        batch_size=batch_size,
        update_freq='epoch',
        write_graph=True,
        write_grads=True
    )
    tensorboard.set_model(model)

    checkpoint_path = './intersavedmodel/'+model_name+'/'+model_name+'_epoch_{epoch}_val_acc_{val_accuracy:.2f}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )

    train_file_nums = list(range(1,num_files_train+1))
    random.shuffle(train_file_nums)
    val_file_nums = list(range(1,num_files_val+1))
    random.shuffle(val_file_nums)
    #Main Training Loop
    print('Starting training')
    train_start = time()
    if 'lstm' in model_name.lower():
        batches_per_file = np.ceil((NUM_SAMPLES_PER_FILE/seq_len)/batch_size)
        history = model.fit_generator(
            generate_batch_seq(train_file_nums,data_dir,seq_len,batch_size)
            , steps_per_epoch = batches_per_file*num_files_train
            , validation_data = generate_batch_seq(val_file_nums,val_data_dir,seq_len,batch_size,train=False)
            , validation_steps = batches_per_file*num_files_val
            , epochs = epochs
            , initial_epoch=initial_epoch
            , verbose = 1
            , shuffle = False
            , callbacks=[tensorboard,checkpoint]
        )
    else:
        batches_per_file = np.ceil(NUM_SAMPLES_PER_FILE/batch_size)
        history = model.fit_generator(
            generate_batch(train_file_nums,data_dir,batch_size,resize=resize)
            , steps_per_epoch = batches_per_file*num_files_train
            , validation_data = generate_batch(val_file_nums,val_data_dir,batch_size,train=False,resize=resize)
            , validation_steps = batches_per_file*num_files_val
            , epochs = epochs
            , initial_epoch=initial_epoch
            , verbose = 1
            , shuffle = False
            , callbacks=[tensorboard,checkpoint]
        )
    train_end = time()
    train_time = train_end - train_start
    print(f'Total Training Time for {epochs} epochs, {num_files_train} files, and batch size {batch_size}: {train_time}')


if __name__=='__main__':
    main()
