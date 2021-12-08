import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tflearn.optimizers import RMSProp
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
import os,sys
from CNN import cnn
from alexnet import alexnet
from alexnetv2 import alexnetv2
from inceptionv3 import inception_v3
# 다른 모든 모델들 넣기
# Self Driving Car algorithms

def main():
    # modelname = sys.argv[1]
    modelname = "AlexNetV2"
    # data_dir = sys.argv[2]
    data_dir = "training_data"
    data = []
    i = 0
    for root,dirs,files in os.walk(data_dir,topdown=False) :
        for file_name in files:
            if i > 1:
                break
            full_path = os.path.join(root,file_name)
            data.extend(np.load(full_path,allow_pickle=True))
            i += 1

    #cv2.imshow("frame",data[0][0])
    #cv2.waitKey(5000)
    data = np.array(data)
    print(np.sum(data[:,0]))
    data = data[2:4,:]
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)

    # Inception V3
    if modelname=="Inceptionv3":
        model=inception_v3()
        history = model.fit(images, labels, 2, validation_set=None)
    else:
        # CNN
        if modelname=="CNN":
            model=cnn()
        
        # AlexNet
        if modelname=="AlexNet":
            model=alexnet()

        if modelname=='AlexNetV2':
            model = alexnetv2()

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

        history = model.fit(images, labels, batch_size=2,epochs=30, validation_data=None)
        print(model.predict(images))
        print(labels)
    #model.save('./test_model.h5')
    #cv2.destroyAllWindows()
if __name__=='__main__':
    main()