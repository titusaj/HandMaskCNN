# Titus John
# 6-27-2017
# Leventhal Lab, University of Michigan
#-----------------------------
#Input
#Input images - these are the raw color images
#Labled images - these are the segmented ground truth images

#Output

#------------------------------
#This scripts is used to import the data into the workspace for data preprocessing

import os
import numpy as np

from skiimaage.io import imsave, imread

data_path ='raw'

image_rows = 420
image_cols = 580

def create_train_data();
    train_data_path = os.path.join(data_path, 'train')
    iamges = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, iamges_rows, image_cols) , dtype=np.uint8)
    image_mask = np.ndarray((total, images_rows, image_cols), dtype=unint.8)

    i = 0
    print('-'*30)
    print('Creating training images')
    print('i'*30)
    for image_name in images
        if 'mask' in image_name;
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        imh = imread(os.path.join(train_data_path, image_name) as_grey = True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i%100 ==0
            print('Done: {0}/{1} images'.format(i, total))
            i += 1
    print('Done loading iamges.')
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy',imgs_mask)
    print('Saving training images to .npy files.')
