# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:51:10 2017

@author: Nicolas
"""

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
import os, cv2, random
import numpy as np

import matplotlib.pyplot as plt

class conv_network():
    
    
    def __init__(self, height, width, optimizer, objective, model_type):
        
        self.height = height
        self.width = width
        self.optimizer = optimizer
        self.objective = objective
        
        self.model_type = model_type
        
        if self.model_type == 'VGG_11':
            
            self.VGG_11()
        
        if self.model_type == 'VGG_16':
        
            self.VGG_16()
            
        if self.model_type == 'VGG_19':
            
            self.VGG_19()
            
        if self.model_type == 'Simple':
            
            self.simple()
            
    def simple(self):
         # réseau de neurone convolutif simple à 6 couches
        
        model = Sequential()
        model.add(Conv2D(64, (3, 3), border_mode='same', input_shape=(3,self.height, self.width), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
        model.add(Conv2D(128, (3, 3), border_mode='same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
        model.add(Conv2D(256, (3, 3), border_mode='same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.compile(loss=self.objective, optimizer=self.optimizer, metrics=['accuracy'])
        
        self.model = model
        
    def VGG_11(self):
         # réseau de neurone convolutif à 11 couches VGG-11
        
       model = Sequential()

       model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(3, self.height, self.width), activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
       model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
       
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
       model.add(Flatten())
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
        
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
    
       model.add(Dense(1))
       model.add(Activation('sigmoid'))
       
       model.compile(loss=self.objective, optimizer=self.optimizer, metrics=['accuracy'])
        
       self.model = model
    
    def VGG_16(self):
        # réseau de neurone convolutif à 16 couches VGG-16
        
       model = Sequential()

       model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(3, self.height, self.width), activation='relu'))
       model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
       model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
       
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
       
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
       model.add(Flatten())
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
        
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
    
       model.add(Dense(1))
       model.add(Activation('sigmoid'))
    
       model.compile(loss=self.objective, optimizer=self.optimizer, metrics=['accuracy'])
        
       self.model = model
       
    def VGG_19(self):
        # réseau de neurone convolutif à 19 couches VGG-19
        
       model = Sequential()

       model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=(3, self.height, self.width), activation='relu'))
       model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
       model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
       
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(Conv2D(512, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
       model.add(Flatten())
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
        
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
    
       model.add(Dense(1))
       model.add(Activation('sigmoid'))
    
       model.compile(loss=self.objective, optimizer=self.optimizer, metrics=['accuracy'])
        
       self.model = model
   
    
    def run(self, X_train, Y_train, X_test, Y_test, batch_size, nb_epoch):
        #Apprentissage et test du modèle
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   
        
        #history = LossHistory()
        history = TestCallback((X_test, Y_test))
        
        self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])   

        self.predictions = self.model.predict(X_test, verbose=0)
        
        loss = history.losses
        val_loss = history.val_losses
        accuracy = history.accuracy
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(self.model_type + ' Loss Trend')
        plt.plot(loss, 'blue', label='Training Loss')
        plt.plot(val_loss, 'green', label='Validation Loss')
        plt.xticks(range(0,nb_epoch)[0::2])
        plt.legend()
        plt.show()
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(self.model_type + ' Accuracy')
        plt.plot(accuracy, 'blue', label='Accuracy')
        plt.xticks(range(0,nb_epoch)[0::2])
        plt.legend()
        plt.show()
        
        for i in range(0,10):
            if self.predictions[i, 0] >= 0.5: 
                print('I am {:.2%} sure this is a Dog'.format(self.predictions[i][0]))
            else: 
                print('I am {:.2%} sure this is a Cat'.format(1-self.predictions[i][0]))
        
            plt.imshow(X_test[i].T)
            plt.show()
            
        print(self.model.summary())
        
        
    def predict(self, path):
    #prediction pour une image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
        
        data = np.ndarray((1, 3, self.height, self.width), dtype=np.uint8)
        data[0] = img.T
        prediction = self.model.predict(data, verbose=0)
        print(prediction)
        if (prediction >= 0.5): 
            print('I am {:.2%} sure this is a Dog'.format(prediction[0][0]))
        else: 
            print('I am {:.2%} sure this is a Cat'.format(1-prediction[0][0]))
            
        plt.imshow(data[0].T)
        plt.show()
        
        
class TestCallback(Callback):
    #Enregistrement des résultats : précision, erreur
    def __init__(self, test_data):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        self.accuracy.append(acc)
        

        