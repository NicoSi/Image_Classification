# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pandas as pd
from Preprocess import data

train_path = 'train'
test_path = 'test'

list_class = ['cats', 'dogs']

height = 150
width = 150
channels = 3

d = data(list_class, train_path, test_path, 1000, 400, height, width, channels)

train_images = d.train_images
test_images = d.train_images

labels = d.train_labels

d.show(200)