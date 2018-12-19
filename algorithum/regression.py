# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_label = np.linspace(1,10,10) + np.random.uniform(-1.5,1.5,10)

y_label = np.linspace(1,10,10) + np.random.uniform(-1.5,1.5,10)

plt.plot(x_label,y_label,'*')
m=tf.Variable(0.44)
b = tf.Variable(0.87)
error =0
for x,y in zip(x_label,y_label):
    y_hat = m*x +b
    error += (y-y_hat)**2 
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)