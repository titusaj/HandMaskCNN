# Titus John
# 7-18-17
# Leventhal Lab, University of Michigan
#-----------------------------
#Input
#Input images - these are the raw color images
#Labled images - these are the segmented ground truth images

#Output

#------------------------------
#This scripts is used to import the data into the workspace for data preprocessing
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Convolution2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


#import the image nparrays
imgs_train = np.load('imgs_train.npy')
imgs_mask_train = np.load('imgs_mask_train.npy')

print(imgs_train.shape)
print(imgs_mask_train.shape)


# dimensions of our images.
img_width, img_height = 471, 441

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mae', 'acc'])

    return model


# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

#load images
seed = 1
batch_size = 16

image_generator = image_datagen.flow_from_directory(
    'dataImages',
    class_mode=None,
    seed=seed)


mask_generator = mask_datagen.flow_from_directory(
    'dataMasks',
    class_mode=None,
    seed=seed)


# Provide the same seed and keyword arguments to the fit and flow methods
image_datagen.fit(imgs_train , augment=True, seed=seed)
mask_datagen.fit(imgs_mask_train , augment=True, seed=seed)

model = get_unet()
print("got here")
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)


model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
          validation_split=0.2,
          callbacks=[model_checkpoint])

model.save_weights('first_try.h5')
