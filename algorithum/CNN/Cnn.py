# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#importing the data set (alredy feature extracted)
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#data pre visualing task
mnist.train.images
mnist.train.num_examples

import matplotlib.pyplot as plt
mnist.train.images.shape
mnist.train.images[2].shape
single_image = mnist.train.images[10].reshape(28,28)
plt.imshow(single_image,cmap='gist_gray')

#tesnorflow 

#Placeholders
x = tf.placeholder(tf.float32,shape=[None,786])
#Varaiable
w = tf.Variable(tf.zeros([786,10]))
b = tf.Variable(tf.zeros([10]))
#Create graph
y= tf.matmul(x,w) +b

#losss function
y_true = tf.placeholder(tf.float32,shape=[None,10])
# cross entropy fuction
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learing_rate=0.5)
train = optimizer.minimize(cross_entropy)

#crate Tensorflow session
init = tf.global_varaiables_initializer()