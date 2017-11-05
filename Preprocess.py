# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:18:10 2017

@author: Nicolas
"""

import os, cv2, random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class data:
   """Classe définissant les données en entré"""
   """ 
       - un chemin d'accès des données d'entrainement
       - un chemin d'accès des données de test
       - une liste de classe (pour l'instant chien et chat)
       - hauteur de l'image
       - largeur de l'image
       - nombre de canaux
       """
   def __init__(self, list_class, train_path, test_path, train_count, test_count, height, width, channels):
       #constructeur
       
        self.list_class = list_class
        self.train_path = train_path
        self.test_path = test_path
        self.height = height
        self.width = width
        self.channels = channels                
        self.train_labels = []
        self.test_labels = []
                
        #liste des chemins vers les images
        self.train_images_class_0 = self.read_image_train(self.train_path, 0)
        self.train_images_class_1 = self.read_image_train(self.train_path, 1)
        self.train_images = self.train_images_class_0[:train_count] + self.train_images_class_1[:train_count]       
        
        seed = 448
        random.seed(seed)
        
        random.shuffle(self.train_images)
        random.shuffle(self.train_labels)
        
        self.test_images_class_0 = self.read_image_test(self.train_path, 0)
        self.test_images_class_1 = self.read_image_test(self.train_path, 1)
        self.test_images = self.test_images_class_0[:test_count] + self.test_images_class_1[:test_count]
        
        random.shuffle(self.test_images)
        random.shuffle(self.test_labels)        
        

        self.train = self.data_process(self.train_images)
        self.test = self.data_process(self.test_images)
        
        print("Train shape: {}".format(self.train.shape))
        print("Test shape: {}".format(self.test.shape))
        
        sns.countplot(self.train_labels)   
        
              
        
   def read_image_train(self, path, cl):
      
       data = []
      
       for element in os.listdir(path + '/' + self.list_class[cl]) :
           
           data.append(path + '/' + self.list_class[cl] + '/' + element)
           self.train_labels.append(cl)
              
       return data
   
   def read_image_test(self, path, cl):
       
       data = []
            
       for element in os.listdir(path + '/' + self.list_class[cl]) :
           
           data.append(path + '/' + self.list_class[cl] + '/' + element)
           self.test_labels.append(cl)
              
       return data
           
        
   def data_process(self, images):
        
        count = len(images)
      
        data = np.ndarray((count, self.channels, self.height, self.width), dtype=np.uint8)
        
        i = 0
        
        for element in images:
            
            img = self.import_image(element)
            data[i] = img.T
            if i%250 == 0: print('Processed {} of {}'.format(i, count))
                
            i = i + 1     
                
        return data
        
   def import_image(self, img):
    #importation et redimensionnement des images
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        return cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
    
    
   def show(self, idx):
       
        image_class_0 = self.import_image(self.train_images_class_0[idx])
        image_class_1 = self.import_image(self.train_images_class_1[idx])
        pair = np.concatenate((image_class_0, image_class_1), axis=1)
        plt.figure(figsize=(10,5))
        plt.imshow(pair)
        plt.show()