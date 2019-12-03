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

train_csv = pd.read_csv('../../../Kaggle_Data/understanding_cloud_organization/train.csv')
sub_csv = pd.read_csv('../../../Kaggle_Data/understanding_cloud_organization/sample_submission.csv')

BASE_DIR = '../../../Kaggle_Data/understanding_cloud_organization/train_images/'

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

def show_RoI(img_mask, img_origin, mask = True):
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
    
    result = img_mask.copy()
    #print(img_mask.shape)
    if mask:
        result[np.where(img_mask==255)] = img_gray[np.where(img_mask==255)]
    else:
        result[np.where(img_mask==0)] = img_gray[np.where(img_mask==0)]
        result[np.where(img_mask==255)] = 0
    return result

def get_binary_image(i, mask= True):
    img = cv2.imread(BASE_DIR + train_df['ImageId'][i])
    img_mask = rle_to_mask(train_df['EncodedPixels'][i], img)
    img_new = show_RoI(img_mask, img, mask)
#    print("Label Type:", train_df['Label'][i])
#    show(img_new)
    return img_new

def get_onehot_label(i):
    label_type = train_df['Label'][i]
    if label_type == "Fish":
        label = [1,0,0,0,0]
    elif label_type == "Flower":
        label = [0,1,0,0,0]
    elif label_type == "Gravel":
        label = [0,0,1,0,0]
    elif label_type == "Sugar":
        label = [0,0,0,1,0]
    return label

def show_class_num(label_array):
    print("-------------- Number of Class Distribution -----------------")
    print("     Fish, Flower, Gravel, Sugar, Non:\n     {0}, {1}, {2}, {3}, {4}".format(
            label_array[:,0].sum(),label_array[:,1].sum(),
            label_array[:,2].sum(),label_array[:,3].sum(),label_array[:,4].sum()))
    print("-------------------------------------------------------------")
    
if __name__ == "__main__":     
    train_csv = train_csv.fillna(-1)
    train_csv['ImageId'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[0])
    train_csv['Label'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[1])
    train_csv = train_csv.drop('Image_Label', axis=1)
    
    print('Total', len(train_csv['ImageId'].unique()),'Images for', 
          len(train_csv['Label'].unique()), 'Types.')
    
    train_df = train_csv[train_csv['EncodedPixels']!=-1].reset_index().drop("index", axis = 1)
    
    data_t = []
    data_f = []
    label_array_t = []    
    label_array_f = []
    pics_num = 300
    print("This generates the dataset with first {0} images".format(pics_num))
    size = 64
    
    print("Generating...")
    # Crop Image Patches for 4 Class
    for num in range(pics_num):
        # Crop Image Patches for 4 Class
        bin_img_t = get_binary_image(num, True)
        # Crop Image Patches for Non Labeled Class
        bin_img_f = get_binary_image(num, False)
        
        height, width = bin_img_t.shape[0], bin_img_f.shape[1]
        
        label_t = get_onehot_label(num)
        label_f = [0,0,0,0,1]
        
        for i in range(int(height/size)):
            for j in range(int(width/size)):
                crop_t = bin_img_t[i*size:(i+1)*size, j*size:(j+1)*size]
                crop_f = bin_img_f[i*size:(i+1)*size, j*size:(j+1)*size]
                if np.any(crop_t == 0) and np.any(crop_f == 0):
                    continue
                elif np.any(crop_t == 0):
                    data_f.append(crop_f)
                    label_array_f.append(label_f)
                elif np.any(crop_f == 0):
                    data_t.append(crop_t)
                    label_array_t.append(label_t) 
        if num%100==0 and num > 0: print("{0} images are done".format(num))
                
    # Merge 4 class label cropped image and non labeled cropped image, euqalizing number of data 
    data = np.concatenate([np.array(data_t), 
                           np.array(data_f)[np.random.choice(len(data_f), int(len(data_t)/4), replace=False), :, :, :]], axis=0)
    label_array = np.concatenate([np.array(label_array_t), 
                                  np.array(label_array_f)[np.random.choice(len(label_array_f), int(len(label_array_t)/4), replace=False), :]], axis=0)
    
    np.save("data{0}_{1}.npy".format(label_array.shape[1],pics_num), data)
    np.save("label{0}_{1}.npy".format(label_array.shape[1],pics_num), label_array)
    print("Finished and saved in npy file\n")
    
    show_class_num(label_array)
    
    
    
    
    
    
    
