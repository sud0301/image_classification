from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import collections

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import tarfile
import os
import math
import time
import sys
import ntpath
from six.moves import urllib
from subprocess import call

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

class DataSet(object):
    def __init__(self, images, labels, one_hot=False, dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)

        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            if images.shape[0] >0 and np.max(images) > 1:
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        # increase the index in epoch by the batch size
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def make_cifar10_dataset(cifar_dir,n_validation = 0,vectorize=False):
    NUM_CLASSES = 10
    NUM_TRAIN   = 50000
    NUM_TEST    = 10000

    cifar10_dir = 'data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # reshape to vectors
    if vectorize:
        X_train = np.reshape(X_train,(X_train.shape[0],-1))
        X_test  = np.reshape(X_test,(X_test.shape[0],-1))

    # make one-hot coding
    y_train_temp = np.zeros((NUM_TRAIN,NUM_CLASSES))
    for i in range(NUM_TRAIN):
        y_train_temp[i,y_train[i]] = 1
    y_train = y_train_temp

    y_test_temp = np.zeros((NUM_TEST,NUM_CLASSES))
    for i in range(NUM_TEST):
        y_test_temp[i,y_test[i]] = 1
    y_test = y_test_temp

    # make validation set
    X_valid = X_train[:n_validation]
    X_train = X_train[n_validation:]

    y_valid = y_train[:n_validation]
    y_train = y_train[n_validation:]

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)

def read_cifar10_dataset(cifar_dir,n_validation = 0,vectorize=False):
    X_train, y_train, X_valid, y_valid, X_test, y_test = make_cifar10_dataset(cifar_dir,n_validation,vectorize)
    train_    = DataSet(X_train, y_train)
    test_     = DataSet(X_test, y_test)
    validate_ = DataSet(X_valid, y_valid)

    return Datasets(train=train_,validation=validate_,test=test_)

def print_data_shapes(datasets):
    print('Training data shape: {}'.format(datasets.train.images.shape))
    print('Training labels shape: {}'.format(datasets.train.labels.shape))
    print('Validation data shape: {}'.format(datasets.validation.images.shape))
    print('Validation labels shape: {}'.format(datasets.validation.labels.shape))
    print('Test data shape: {}'.format(datasets.test.images.shape))
    print('Test labels shape: {}'.format(datasets.test.labels.shape))

def get_time_stamp():
    date_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def maybe_download(data_dir):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        call(["tar", "-xvzf", './data/cifar-10-python.tar.gz', '-C', data_dir ]) 
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y
 
def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        print (f) 
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte 
