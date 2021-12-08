import numpy as np
import random
import os
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
import cv2

def preprocess_data_seq(data):
    images = []
    labels = []
    for seq in data:
        images.append(np.array([f[0] for f in seq],dtype=np.float))
        labels.append(np.array(seq[-1,1],dtype=np.int))
    images = np.stack(images)
    images /= 255.0
    #images = preprocess_input(images)
    labels = np.stack(labels)
    return images,labels

def preprocess_data(data,resize=False):

    data = np.array(data)
    #images = preprocess_input(np.array(list(data[:,0]),dtype=np.float))
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)
    if resize:
        scale_percent = 120 # percent of original size
        width = int(IMAGE_WIDTH * scale_percent / 100)
        height = int(IMAGE_HEIGHT * scale_percent / 100)
        dim = (width, height)
        resized_images = []
        for img in list(data[:,0] / 255.0):
            resized_images.append(cv2.resize(img, dim, interpolation = cv2.INTER_AREA))
        resized_images = np.array(resized_images)
        images = resized_images
    #labels = np.argmax(labels, axis=1)
    return images,labels

def generate_batch(file_nums,data_dir,batch_size,train=True,resize=False):

    while True:

        if train:
            random.shuffle(file_nums)
        for file_num in file_nums:
            if train and file_num==7:
                continue
            file_path = os.path.join(data_dir,f"training_data-{file_num}.npy")
            data = np.load(file_path,allow_pickle=True)
            if train:
                random.shuffle(data)
            batch_start = 0
            while batch_start < len(data):
                batch_data = data[batch_start:batch_start+batch_size]
                batch_start += batch_size
                yield preprocess_data(batch_data,resize)


def generate_batch_seq(file_nums,data_dir,seq_len,batch_size,train=True):

    while True:

        if train:
            random.shuffle(file_nums)
        for file_num in file_nums:
            if train and file_num==7:
                continue
            file_path = os.path.join(data_dir,f"training_data-{file_num}.npy")
            data = np.load(file_path,allow_pickle=True)
            samples = []
            sample_start = 0
            while sample_start < len(data)-seq_len:
                sample_data = data[sample_start:sample_start+seq_len]
                sample_start += 1
                samples.append(sample_data)
            if train:
                random.shuffle(samples)
            samples = np.array(samples)
            batch_start = 0
            while batch_start < len(samples):
                batch_data = samples[batch_start:batch_start+batch_size]
                batch_start += batch_size
                yield preprocess_data_seq(batch_data)