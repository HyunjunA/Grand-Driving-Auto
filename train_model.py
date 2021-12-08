import numpy as np
import argparse
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
import os,sys
import os.path
from os import path
from CNN import cnn
from alexnet import alexnet
from alexnetv2 import alexnetv2
from xception import xception
#from inceptionv3 import inception_v3
from inceptionv3Keras import InceptionV3
from resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import preprocess_input
#from sklearn.utils.class_weight import compute_class_weight
#from sklearn.model_selection import KFold
from time import time
import random
import datetime


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self,step):
        return
def preprocess_data(data):

    data = np.array(data)
    #images = preprocess_input(np.array(list(data[:,0]),dtype=np.float))
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)
    #labels = np.argmax(labels, axis=1)
    return images,labels

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
    parser.add_argument('--num_files','-n',type=int)
    parser.add_argument('--model_name','-m',type=str,nargs='?',default='AlexNetV2')
    parser.add_argument('--epochs','-e',type=int,nargs='?',default=10)
    parser.add_argument('--batch_size','-b',type=int,nargs='?',default=32)
    parser.add_argument('--learning_rate','-lr',type=float,nargs='?',default=0.0001)

    # Training parameters
    args = parser.parse_args()
    model_name = args.model_name
    data_dir = args.data_dir
    num_files = args.num_files
    epochs = args.epochs
    batch_size = args.batch_size
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
    if model_name=="CNN":
        model = cnn()
    elif model_name== "InceptionV3":
        model = InceptionV3()
    elif model_name=="AlexNet":
        model = alexnet()
    elif model_name=="AlexNetV2":
        model = alexnetv2()
    elif model_name== "Xception":
        model = xception()
    elif model_name== "ResNet50":
        model = ResNet50()
    # else:
    #     # TFlearn InceptionV3
    #     model = inception_v3()
        

    # Load saved model if it exists
    root_saved_model_local = os.getcwd()+"/intersavedmodel/"
    root_saved_model_docker = "./intersavedmodel/"
    saved_model_file_name = "test_model_"+model_name+"_epochs"+"_"+str(epochs)+"_batchsize_"+str(batch_size)+".h5"
    saved_model_docker = path.join(root_saved_model_docker,saved_model_file_name)
    saved_model_local = path.join(root_saved_model_local,saved_model_file_name)

    if path.exists(saved_model_local):
        print("Load saved model file before training")
        #model=load_model(saved_model_local)
    elif path.exists(saved_model_docker):
        print("Load saved model file before training")
        #model=load_model(saved_model_docker)
    else:
        print("Cannot load saved model file before training")

    #For tflearn version
    #if model_name == 'InceptionV3':
        #acc = Accuracy()
        #network = regression(model, optimizer='momentum',
                             #loss='categorical_crossentropy',
                             #learning_rate=learning_rate, name='targets', metric=acc)
        #model = tflearn.DNN(network, max_checkpoints=0, tensorboard_verbose=0,tensorboard_dir='log')
    #else:
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./tb_logs',
        histogram_freq=0,
        batch_size=batch_size,
        update_freq='batch',
        write_graph=True,
        write_grads=True
    )
    tensorboard.set_model(model)

    #Main Training Loop
    print('Starting training')
    train_start = time()
    test_metrics = {'val_loss':0.0,'val_accuracy':0.0}
    for e in range(epochs):
        print(f'Epoch {e}:')
        print('---------------------------------------------------------------------------------------------------------')
        #Get list of all file numbers, shuffle them
        file_nums = list(range(1,num_files+1))
        random.shuffle(file_nums)
        i = 0
        #if e > 0 and e % 2==0 :
            #K.set_value(model.optimizer.learning_rate,learning_rate/5)
            #learning_rate /= 5
        #iterate through all data
        batch_no = 1
        while i < len(file_nums):
            data = []

            # Load 5 files
            for file_num in file_nums[i:i+5]:
                if file_num==7:
                    continue
                file_path = os.path.join(data_dir,f"training_data-{file_num}.npy")
                file_data = np.load(file_path,allow_pickle=True)
                data.extend(file_data)
            #Split into train and test
            train_split = int(len(data)*0.8)
            train = data[:train_split]
            test = data[train_split:]
            del data
            i += 5

            # To solve imbalanced data problem
            # class_weights = class_weight.compute_class_weight('balanced',
            #                                              np.unique(labels),
            #                                              labels)
            # class_weights = class_weight.compute_class_weight('balanced',
            #                                              class_labels,
            #                                              labels)
            # sample_weights = class_weight.compute_sample_weight(class_weight,labels)
            # y_integers = np.argmax(labels, axis=1)
            # class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
            # d_class_weights = dict(enumerate(class_weights))

            batch_start = 0
            #Generate batches from train dataset
            while batch_start < len(train):
                batch_data = train[batch_start:batch_start+batch_size]
                batch_start += batch_size
                X_train,y_train = preprocess_data(batch_data)
                #For tflearn version
                #if model_name == 'InceptionV3':
                    #train_metrics = model.fit_batch({'input':X_train},{'targets':y_train})
                    #train_metrics = {'loss':train_metrics}
                #else :
                train_metrics = model.train_on_batch(X_train, y_train,reset_metrics=False)
                train_metrics = {'train_loss':train_metrics[0],'train_accuracy':train_metrics[1]}
                tensorboard.on_batch_end(batch_no, train_metrics)
                if batch_start < len(train):
                    tensorboard.on_batch_end(batch_no, test_metrics)
                batch_no += 1

            # Eval after training on 5 files
            batch_start = 0
            while batch_start < len(test):
                batch_data = test[batch_start:batch_start+batch_size]
                batch_start += batch_size
                X_test,y_test = preprocess_data(batch_data)
                test_metrics = model.test_on_batch(X_test,y_test,reset_metrics=False)
            test_metrics = {'val_loss':test_metrics[0],'val_accuracy':test_metrics[1]}
            tensorboard.on_batch_end(batch_no,test_metrics)
            #For tf learn version
            #if model_name == 'InceptionV3':
                #test_metrics = model.evaluate(X_test,y_test)
                #test_metrics = {'accuracy':test_metrics}
            #else:
            #test_metrics = model.test_on_batch(X_test,y_test,reset_metrics=False)
            #test_metrics = {'val_loss':test_metrics[0],'val_accuracy':test_metrics[1]}
            #tensorboard.on_batch_end(batch_no, test_metrics)
            print(f'Train metrics after 5 files: {train_metrics}  Test metrics after 5 files: {test_metrics}')

            # Save model every 40 files
            if i % 40 == 0:
                print('Saving Model')
                hfivename='./intersavedmodel/test_model_'+model_name+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
                #model.save(hfivename)

        # Save model at the end of each epoch
        print(f'End epoch {e}, saving model')
        hfivename='./intersavedmodel/test_model_'+model_name+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
        model.save(hfivename)

    # Print final metrics
    print('Training finished!')
    print('Final training metrics:')
    print(train_metrics)
    print('Final test metrics:')
    print(test_metrics)
    train_end = time()

    # Final save model
    hfivename='./intersavedmodel/test_model_'+model_name+'_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.h5'
    model.save(hfivename)
    print('Model Saved')

    train_time = str(datetime.timedelta(seconds=train_end-train_start))
    print(f'Total Training Time for {epochs} epochs, {num_files} files, and batch size {batch_size}: {train_time}')


if __name__=='__main__':
    main()
