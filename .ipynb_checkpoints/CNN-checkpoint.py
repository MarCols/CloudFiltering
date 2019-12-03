# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:58:42 2019

@author: Kenichi
"""

import os
import numpy as np
import pandas as pd
import cv2
import random
from matplotlib import pyplot as plt

train_csv = pd.read_csv('../../Kaggle_Data/understanding_cloud_organization/train.csv')
sub_csv = pd.read_csv('../../Kaggle_Data/understanding_cloud_organization/sample_submission.csv')

BASE_DIR = '../../Kaggle_Data/understanding_cloud_organization/train_images/'

def show(img):
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
def rle_to_mask(rle_string, img):
    
    rows, cols = img.shape[0], img.shape[1]
    img = np.zeros(rows*cols, dtype=np.uint8)

    rle_numbers = [int(x) for x in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
#    print(rle_pairs)
    for index, length in rle_pairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    img = np.expand_dims(img, axis=2)
    return img

def ignore_background(img_mask, img_origin):
    assert img_mask.shape == img_mask.shape
    
    result = img_mask.copy()
    result[np.where(img_mask==255)] = img_origin[np.where(img_mask==255)]
    
    return result

def get_binary_image(i):
    img = cv2.imread(BASE_DIR + train_df['ImageId'][i])
    img_mask = rle_to_mask(train_df['EncodedPixels'][i], img)
    img_new = ignore_background(img_mask, img)
    print("Label Type:", train_df['Label'][i])
    show(img_new)
    return img_new


train_csv = train_csv.fillna(-1)
train_csv['ImageId'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[0])
train_csv['Label'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[1])
train_csv = train_csv.drop('Image_Label', axis=1)

print('Total', len(train_csv['ImageId'].unique()),'Images for', 
      len(train_csv['Label'].unique()), 'Types.')


train_df = train_csv[train_csv['EncodedPixels']!=-1].reset_index().drop("index", axis = 1)
img = cv2.imread(BASE_DIR + train_df['ImageId'][0])
img_mask = rle_to_mask(train_df['EncodedPixels'][0], img)
img_new = ignore_background(img_mask, img)

















