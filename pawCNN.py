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
from keras.layers import Activation, Dropout, Flatten, Dense
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
img_rows, img_cols = 512, 512

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
