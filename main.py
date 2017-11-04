# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:58:07 2017

@author: Nicolas
"""
import tensorflow as tf

a = tf.truncated_normal([16,128,128,3])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.shape(a))