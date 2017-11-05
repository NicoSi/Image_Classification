# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pandas as pd
from Preprocess import data
from Model import conv_network
from keras.optimizers import RMSprop

train_path = 'train'
test_path = 'test'

list_class = ['cats', 'dogs']

#Paramètres de redimensionnement d'images
height = 64
width = 64
channels = 3

#Création des données d'entrainement et de test
d = data(list_class, train_path, test_path, 2000, 500, height, width, channels)

x_test = d.test_images
y_test = d.Y_test

d.show(200)

#Paramètre des modèles
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

nb_epoch = 15
batch_size = 16

#Entrainement et test du modèle
vgg_11 = conv_network(height, width, optimizer, objective, 'VGG_11')
vgg_11.run(d.X_train, d.Y_train, d.X_test, d.Y_test, batch_size, nb_epoch)
vgg_11.predict('test_complementaire/dog.3.jpg')


"""
#Test d'un réseau de neurone convolutif modèle simple
simple_model = conv_network(height, width, optimizer, objective, 'Simple')
simple_model.run(d.X_train, d.Y_train, d.X_test, d.Y_test, batch_size, nb_epoch)

#Test d'un réseau de neurone convolutif à 11 couches VGG-11
vgg_11 = conv_network(height, width, optimizer, objective, 'VGG_11')
vgg_11.run(d.X_train, d.Y_train, d.X_test, d.Y_test, batch_size, nb_epoch)

#Test d'un réseau de neurone convolutif à 19 couches VGG-19
vgg_19 = conv_network(height, width, optimizer, objective, 'VGG_19')
vgg_19.run(d.X_train, d.Y_train, d.X_test, d.Y_test, batch_size, nb_epoch)

vgg_16 = conv_network(height, width, optimizer, objective, 'VGG_16')
vgg_16.run(d.X_train, d.Y_train, d.X_test, d.Y_test, batch_size, nb_epoch)

predictions = conv_model.predictions
"""