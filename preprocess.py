import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import helpers
from sklearn.model_selection import train_test_split
#import wavelet
import pywt

local_path = os.path.dirname(os.path.abspath(__file__))
input_path = local_path + '/understanding_cloud_organization'
print(os.listdir(input_path))


def view_training_set():
    train = pd.read_csv(f'{input_path}/train.csv')
    sub = pd.read_csv(f'{input_path}/sample_submission.csv')
    print(train.head())

    n_train = len(os.listdir(f'{input_path}/train_images'))
    n_test = len(os.listdir(f'{input_path}/test_images'))
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')

    # So we have ~5.5k images in train dataset and they can have up to 4 masks: Fish, Flower, Gravel and Sugar.
    train_labels = train['Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
    print("\n", train_labels)
    mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[1]).value_counts()
    print("\n", mask_count)
    # But there are a lot of empty masks. In fact only 266 images have all four masks. It is important to remember this.
    non_empty_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts().value_counts()
    print("\n", non_empty_mask_count)

    # Get image labels
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    # Plot 4 random figures of each type
    RGB_cases = ["im", "red", "green", "blue"]
    img_mask_dict = dict()
    for case in RGB_cases:
        plt.figure(case, figsize=(25, 16))
    for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):  # columns
        print("im_id=", im_id)
        for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):  # rows
            for case in RGB_cases:
                fig = plt.figure(case)
                ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])
                im = Image.open(f"{input_path}/train_images/{row['Image_Label'].split('_')[0]}")
                red, green, blue = im.split()
                img = eval(case)
                plt.imshow(img)
                #pix = np.array(img)
                could_label = row['label']
                mask_rle = row['EncodedPixels']
                try:  # label might not be there!
                    mask = helpers.rle_decode(mask_rle)
                    img_mask_dict[im_id] = [im, mask]  # store pixels array only if there's mask
                except:
                    mask = np.zeros((1400, 2100))
                plt.imshow(mask, alpha=0.5, cmap='gray')
                ax.set_title(f"Image: {im_id}. Label: {could_label}.", fontsize=10)
    for key, vals in img_mask_dict.items():
        plot_wavelet_results(label=key, img=vals[0], mask=vals[1])


def plot_wavelet_results(label, img, mask):
    red, green, blue = img.split()
    rgb_list = ['red', 'green', 'blue']
    #print(np.array(img).shape)
    #print(mask.shape)
    # Wavelet transform of image, and plot approximation
    fig = plt.figure(label, figsize=(12, 3))
    for i, case in enumerate(rgb_list):
        # Apply wavelet
        img_array = eval(case)
        coeffs2 = pywt.dwt2(img_array, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        # Plot subfigure
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
        #plt.imshow(mask, alpha=0.4, cmap='gray')
        ax.set_title(case, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


def prepare_data_for_modelling(train, sub):
    '''Create a list of unique image ids and the count of masks for images.
    This will allow us to make a stratified split based on this count.'''

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts(). \
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})

    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42,
                                            stratify=id_mask_count['count'], test_size=0.1)

    print("train_ids = ", train_ids)
    print(len(train_ids))
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
    print("len(test_ids) = ", len(test_ids))

    for i in range(0, 10):
        #image_name = '8242ba0.jpg'
        image_name = train_ids[i]
        image = helpers.get_img(image_name, input_path=input_path)
        mask = helpers.make_mask(train, image_name)
        helpers.visualize(image, mask)


if __name__ == "__main__":
    view_training_set()
    plt.show()

