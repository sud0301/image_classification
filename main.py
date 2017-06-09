import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
#import dataset
import random
import math

from data_utils import *
from model_cifar10 import *

folder_data = './data/cifar-10-batches-py'

n_batch = 100
n_epoch = 200
folder_data = './data/'
maybe_download(folder_data)
imagesize = 32
n_channel= 3

cifar10 = read_cifar10_dataset(folder_data)

'''
Placeholders
Placeholder variables serve as the input to the TensorFlow computational graph that we may change each time we execute the graph.
'''

# This is the placeholder for the input images. The convolutional layers expect the input_x tensor to be 4-dim tensor [num_image, image_height, image_width, num_channel]
input_x = tf.placeholder(tf.float32, shape=[None, imagesize, imagesize, n_channel])

# This is the placeholder variable for the true labels. The shape of this variable is [num_image, num_classes]. It expects a one-hot vector. 
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
# This calculates the class index
y_true_cls = tf.argmax(y_true, dimension=1)

# This is the placeholder for the learning mode. [Training  or Testing]
phase_train = tf.placeholder(tf.bool, name = 'phase_train')

# This is the placeholder for the keep probability for Dropout layer.
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

#  This variable loads the output from the Convolutional Network. 
y_pred =  load_model(input_x, keep_prob, phase_train)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true) # try changin logits to y_pred
cost = tf.reduce_mean(cross_entropy)

#Since we have the cost measure, we want to minimize the cost. In this case we use AdamOptimizer, which is a advanced version of Gradient Descent. 
optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Once the tensorflow graph has been created, we have create a session to execute the graph
session = tf.Session()
# Now the variables including weights and biases are initialized.
session.run(tf.global_variables_initializer())

for epoch in range(n_epoch):
    summ = 0
    for idx in range(0, 50000, n_batch):
        x, index_labels = cifar10.train.next_batch(n_batch)
        # Now the input batch is put into a dict with names of the placeholder variables in the tensorflow graph.
        feed_dict_train = {input_x: x, y_true: index_labels, phase_train:True, keep_prob: 0.75}
        # Run the optimizer. Tensorflow assigns variables in feed_dict to the placeholder varaibles. 
        lcnn, _ = session.run([cost, optimizer], feed_dict = feed_dict_train)
        
    loss_, acc  = session.run([cost, accuracy], feed_dict_train)
    print ("Epoch: " + str(epoch) + ",Minibatch Loss= " + "{:.6f}".format(loss_) + ",Training Accuracy= " + "{:.5f}".format(acc))

    for idx in range(0, 10000, n_batch):
        x, index_labels = cifar10.test.next_batch(n_batch)
        feed_dict_test = {input_x: x, y_true:index_labels, phase_train:False, keep_prob:1.0} 
        loss_, acc  = session.run([cost, accuracy], feed_dict_test)
        summ = summ + acc
    summ = (summ*n_batch)/10000
    print ('Testing Accuracy= ' + '{:.5f}'.format(summ))
