""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.
Applying shortened VGG (13-layers) convolutional network to cats and dogs kaggle
Dataset classification task.
References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
    http://arxiv.org/pdf/1409.1556
"""
from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.activations import relu
from tflearn.layers.normalization import batch_normalization
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

# Data loading and preprocessing

files_path = 'path_to_the_training_images'

cat_files_path = os.path.join(files_path, 'cat*.jpg')
dog_files_path = os.path.join(files_path, 'dog*.jpg')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

n_files = len(cat_files) + len(dog_files)
print(n_files)

size_image = 128

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0
for f in cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue

for f in dog_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue
   

# Prepare train & test samples
###################################

# test-train split   
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)


###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 128 x 128 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, size_image, size_image, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Building 'VGG Network'

''' Batch normalization is additionally used here '''

network = relu(batch_normalization(conv_2d(network, 64, 3, activation=None, bias = False)))
network = relu(batch_normalization(conv_2d(network, 64, 3, activation=None, bias = False)))
network = max_pool_2d(network, 2, strides=2)

network = relu(batch_normalization(conv_2d(network, 128, 3, activation=None, bias = False)))
network = relu(batch_normalization(conv_2d(network, 128, 3, activation=None, bias = False)))
network = max_pool_2d(network, 2, strides=2)

network = relu(batch_normalization(conv_2d(network, 256, 3, activation=None, bias = False)))
network = relu(batch_normalization(conv_2d(network, 256, 3, activation=None, bias = False)))
network = relu(batch_normalization(conv_2d(network, 256, 3, activation=None, bias = False)))
network = max_pool_2d(network, 2, strides=2)

network = relu(batch_normalization(conv_2d(network, 512, 3, activation=None, bias = False)))
network = relu(batch_normalization(conv_2d(network, 512, 3, activation=None, bias = False)))
network = relu(batch_normalization(conv_2d(network, 512, 3, activation=None, bias = False)))
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg_cats_and_dogs',
                    max_checkpoints=1, tensorboard_verbose=0)
model.fit(X, Y,validation_set=(X_test, Y_test), n_epoch=100, shuffle=True,
          show_metric=True, batch_size=32, run_id='vgg_cats_dogs')
model.save('Vgg_cat_dog_6_final.tflearn')
