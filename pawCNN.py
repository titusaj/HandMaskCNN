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
