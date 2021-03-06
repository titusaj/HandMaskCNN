# Titus John
# 6-27-2017
# Leventhal Lab, University of Michigan
#-----------------------------
#Input
#Input images - these are the raw color images
#Labled images - these are the segmented ground truth images

#Output

#------------------------------
#This is an image classifier trained utilizing for segmentation classification
from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from TAJ_DataImport import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256
img_channels = 3

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)




def get_unet():
    inputs = Input((img_rows, img_cols,3))
    print('-'*30)
    print('Input Shape')
    print(inputs)
    print('-'*30)



 #Comennting out this current model
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv1)
    print('-'*30)
    print('Conv 1 Shape')
    print(conv1.shape)
    print('-'*30)
    pool1 = MaxPooling2D(pool_size=(2, 2) , data_format = 'channels_last')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format = 'channels_last')(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),  data_format = 'channels_last')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format = 'channels_last')(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format="channels_last")(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model




def preprocess_color(imgs):
    print('Preprocessing color imags')
    #print(imgs.shape)
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, img_channels), dtype=np.uint8)
    #print (imgs_p.shape)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols, img_channels), preserve_range=True)
    #    print(imgs_p[i].shape)
    #imgs_p = imgs_p[..., np.newaxis] #not sure what this is doing no new axis
    return imgs_p

def preprocess_mask(imgs):
    print('Preprocessing mask images')
    #print(imgs.shape)
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    print (imgs_p.shape)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
    #    print(imgs_p[i].shape)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()



    imgs_train = preprocess_color(imgs_train)
    imgs_mask_train = preprocess_color(imgs_mask_train)


    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)


    print('-'*30)
    print('image train and mask shape...')
    print(imgs_train.shape)
    print(imgs_mask_train.shape)
    print('-'*30)


    model.fit(imgs_train, imgs_mask_train, batch_size=8, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()
