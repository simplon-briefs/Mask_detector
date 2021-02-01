import cv2
import os
import random

import numpy as np

from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras  import Sequential
from keras.layers import Dense



class Keras_mask_detector:
    def __init__(self):
        self.data = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        self.img_to_data()
        self.train_model()


        
    def img_to_data(self):
        categories = ["with_mask", "without_mask"]
        for category in categories:
            path = os.path.join('Dataset_masks/train', category)
    
            label = categories.index(category)
            for file in os.listdir(path):
                img_path = os.path.join(path,file)
                img = cv2.imread(img_path)
                img = cv2.resize(img,(224, 224))
                
                self.data.append([img,label])
        random.shuffle(self.data)
        X = []
        y = []
        for features, label in self.data:
            X.append(features)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.2)

    def train_model(self):
        vgg = VGG16()
        self.model = Sequential()
        for layer in vgg.layers[:-1]:
            self.model.add(layer)
        for layer in self.model.layers:
            layer.trainable=False
        self.model.add(Dense(1,activation='sigmoid'))

        self.model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
        self.model.fit(self.X_train, self.y_train, epochs=5, validation_data=(self.X_test, self.y_test))
        self.model.save('model/model.h5',save_format='h5')

