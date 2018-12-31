import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#importing the data set (alredy feature extracted)
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#============= MakingHELPER FUNCTION ---------------------------
# random intialstion

def init_random(shape):
    random_data_init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Varaiable(random_data_init)

#making basies
def init_biase(shape):
    biase_init = tf.constant(0.1,shape=shape)
    return biase_init

# convultion 
def Con2D(x,w):
    #x ===> [batch,Height,Width,channel(It is ither grey-scale or can be colored and ca be multiple)]
    #w ===> [height filter,width filter,Channel IN, channel Out]
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#maxpollong 
def max_pol2X2(x):
    #x =========> same x
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# convulation layers
def convultionn_layer(input_x,shape):
    weight = init_random(shape)
    baise = init_biase([shape[3]])
    return tf.nn.relu(Con2D(input_x,weight)+baise)


    
    
