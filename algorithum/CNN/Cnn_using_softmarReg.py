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
x = tf.placeholder(tf.float32,shape=[None,784])
#Varaiable
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#Create graph
y= tf.matmul(x,w) +b

#losss function
y_true = tf.placeholder(tf.float32,shape=[None,10])
# cross entropy fuction
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

#crate Tensorflow session
init = tf.global_variables_initializer()

with tf.Session() as sass:
    sass.run(init)
    for step in range(1000):
        # most tricky part of the question
        batch_x,batch_y = mnist.train.next_batch(100)
        sass.run(train,feed_dict={x:batch_x,y_true:batch_y})
        
    #evaluating the model
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    #converting boolen to float
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print(sass.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
    