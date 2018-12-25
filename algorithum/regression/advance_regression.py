# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

x_data = np.linspace(0,10,100000)
noise = np.random.randn(len(x_data))

#y=mx+c
m =0.5
c =5
y_true = m*x_data + c + noise

x_df = pd.DataFrame(data = x_data,columns = ['X_data'])
y_df = pd.DataFrame(data = y_true,columns=['Y_data'])

my_data = pd.concat([x_df,y_df],axis=1)

my_data.sample(n=300).plot(kind ='scatter',x='X_data',y='Y_data')

batch_size = 10

m = tf.Variable(0.81)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = xph*m + b
error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sass:
    sass.run(init)
    batch_siz=1000
    for i in range(batch_siz):
        rand_int = np.random.randint(len(x_data),size=batch_size)
        feed = {xph:x_data[rand_int],yph:y_true[rand_int]}
        sass.run(train,feed_dict=feed)
    model_m , model_n = sass.run([m,b])


y_hat = model_m*x_data + model_n

my_data.sample(n=300).plot(kind='scatter',x='X_data',y='Y_data')
plt.plot(x_data,y_hat,'r')




    
    