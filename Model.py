# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:51:10 2017

@author: Nicolas
"""

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

class NeuralNetworkModel():
    
    
    def __init__(self, height, width, optimizer, objective):
        
        self.height = height
        self.width = width
        self.optimizer = optimizer
        self.objective = objective
    
    def model(self):

       model = Sequential()

       model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, self.height, self.width), activation='relu'))
       model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
    
       model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
       model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
        
       model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
       model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
        
       model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
       #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
    
        #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        #     model.add(MaxPooling2D(pool_size=(2, 2)))
    
       model.add(Flatten())
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
        
       model.add(Dense(256, activation='relu'))
       model.add(Dropout(0.5))
    
       model.add(Dense(1))
       model.add(Activation('sigmoid'))
    
       model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
        
       return model