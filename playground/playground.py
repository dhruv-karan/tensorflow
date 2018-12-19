# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:49:41 2018

@author: dhruv
"""
import tensorflow as tf

a = tf.constant('dhruv')
type(a)
a
b = tf.constant('karan')

with tf.Session() as sass:
    result = sass.run(a+" "+ b)

result

const = tf.constant(10)
fill_mat = tf.fill((4,4),10)
myzeroes = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random_normal((4,4),mean=0,stddev=1.0)
myrandu = tf.random_uniform((4,4),minval=0,maxval=1.0) 

my_ops = [const, fill_mat,myzeroes,myones,myrandn,myrandu]

sess = tf.InteractiveSession()

for op in my_ops:
    print(sess.run(op))

######  variable and plsceholder
my_tensor = tf.random_uniform((4,4),0,1)
my_var = tf.Variable(initial_value=my_tensor)
print(my_var)

init = tf.global_variables_initializer()
sess.run(init)

my_var.eval()

#============= 
import numpy as np
rand_a = np.random.uniform(0,100,(4,4))
rand_b = np.random.uniform(0,50,(4,4))

a = tf.placeholder(tf.float32)
b= tf.placeholder(tf.float32)

op_add = a+b
op_mul = a*b

with tf.Session() as ses:
    add_result = ses.run(op_add,feed_dict={a:rand_a,b:rand_b})
    print(add_result)
    print("\n")
    mul_result = ses.run(op_mul,feed_dict={a:rand_a,b:rand_b})
    print(mul_result)

#================ making psudo nutral network =============
n_features = 10
dense_layer = 3
X = tf.placeholder(tf.float32,(None, n_features))
W = tf.Variable(tf.random_normal([n_features,dense_layer]))
b = tf.Variable(tf.ones([dense_layer]))
WX = tf.matmul(X,W)
z = tf.add(WX,b)
a = tf.sigmoid(z)

iinit = tf.global_variables_initializer()

with tf.Session() as ii:
    ii.run(iinit)
    layer_out = ii.run(a,feed_dict={X: np.random.random([1,n_features])})
    print(layer_out)
