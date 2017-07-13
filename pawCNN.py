# Titus John
# 6-27-2017
# Leventhal Lab, University of Michigan
#-----------------------------
#Input
#Input images - these are the raw color images
#Labled images - these are the segmented ground truth images

#Output

#------------------------------
# This is an image classifier trained utilizing for segmentation classification
# Feeding a color image into the network for training and classification

from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K #this sets the model to use tensorflow as the backend

from paw_DataImport import load_train_data, load_test_data #this imports the test and training images

#TF dimension ordering se in this code
K.set_image_data_format('channels_last')

# Image resizing: Accodring to GitHub forum the image thats is fed into the CNN HandMaskCNN
# has to have dimmesion that are divisible by 16: going to intially try large resized i
#i mage in the training set

img_row = 96
img_cols - 96

#smooth  for dice
smooth = 1.

#the dice cefficent calculation
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true) #flatten the tensor
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return(2. * intersection +smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# loass function of the dice coefficent
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#Setup the neural network
def get_unet()
    inputs = Input(shape=(3, 256, 256))#this sets up a color image to be read
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
