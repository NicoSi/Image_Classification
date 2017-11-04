# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:25:13 2017

@author: Nicolas
"""

import tensorflow as tf
node1 = tf.constant(3, tf.float32)
node2 = tf.constant(4) # also tf.float32 implicitly
print(node1, node2)