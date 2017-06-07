import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
#import dataset
import random
import math

from aae_helpers import *
from data_utils import * 
from model_cifar10 import *

folder_data = './data/cifar-10-batches-py'

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
'''
def convert2one_hot(index_labels):
    one_hot_label = np.zeros((len(index_labels), 10))
    for idx, i in enumerate(index_labels):  
        print (idx, i)
        one_hot_label[idx, i] = 1.0
    return one_hot_label
'''
cifar10 = read_cifar10_dataset(folder_data)

n_batch = 100
n_epoch = 200
folder_data = './data/'
maybe_download(folder_data)
imagesize = 32
n_channel= 3

input_x = tf.placeholder(tf.float32, shape=[None, imagesize, imagesize, n_channel])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

phase_train = tf.placeholder(tf.bool, name = 'phase_train')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

y_pred =  load_model(input_x, keep_prob, phase_train)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true) # try changin logits to y_pred
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(n_epoch):
    summ = 0
    for idx in range(0, 50000, n_batch):
        x, index_labels = cifar10.train.next_batch(n_batch)
        #one_hot_label = convert2one_hot(index_labels) 
        feed_dict = {input_x: x, y_true: index_labels, phase_train:True, keep_prob: 0.75}
        lcnn, _ = session.run([cost, optimizer], feed_dict = feed_dict)
        
    loss_, acc  = session.run([cost, accuracy], feed_dict)
    print ("Epoch: " + str(epoch) + ",Minibatch Loss= " + "{:.6f}".format(loss_) + ",Training Accuracy= " + "{:.5f}".format(acc))

    for idx in range(0, 10000, n_batch):
        x, index_labels = cifar10.test.next_batch(n_batch)
        feed_dict={input_x: x, y_true:index_labels, phase_train:False, keep_prob:1.0} 
        loss_, acc  = session.run([cost, accuracy], feed_dict)
        summ = summ + acc
    summ = summ*n_batch/10000
    print ('Testing Accuracy= ' + '{:.5f}'.format(summ))
