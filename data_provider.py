import os
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from const import class_names
import cv2
import numpy as np
import random
from const import data_path, image_size


class DataProvider:
    '''
    DataProvider class is responsible for providing preprocessed data for classification model. 
    Takes [image_dim] and [train_split] parameters. 
    [image_dim] - (int) dimension of target images (original ones are going to be resized) - the target size of image is [image_dim]x[image_dim]
    [train_split] - (float) specifies ratio between train and test data. 
    If [train_split] = 0.7 it means the training data will consist of 70% images of the whole dataset
    '''

    def __init__(self, image_dim, train_split):
        super().__init__()
        self.image_dim = image_dim
        self.image_size = (image_dim, image_dim)
        self.train_split = train_split

    def get_normalized_data(self, verbose=False):
        '''
        This method returns two tuples, (train_images, train_labels), (test_images, test_labels)
        with normalized images converted to pixel numpy arrays. Data comes from path specified in const.py as data_path
        [verbose] is an optional argument which specifies whether to print more information
        '''
        self.class_names = []
        class_names_dict = []
        data = []

        print('Iterating through data...')
        i = 0
        for class_name in os.listdir(data_path):
            class_names_dict.append({class_name[10:]: i})
            self.class_names.append(class_name[10:])
            if os.path.isdir(f'{data_path}/{class_name}'):
                for image in os.listdir(f'{data_path}/{class_name}'):
                    image_array = cv2.imread(f'{data_path}/{class_name}/{image}')
                    image_array = cv2.resize(image_array, self.image_size)
                    data.append((image_array, i))
            i += 1

        random.shuffle(data)
        
        print('Got data!')
        
        train_size = int(self.train_split * len(data))
        test_size = len(data) - train_size

        train_data = data[:train_size]
        test_data = data[train_size:]

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for (image_array, class_label) in train_data:
            train_images.append(image_array)
            train_labels.append(np.array(class_label))

        for (image_array, class_label) in test_data:
            test_images.append(image_array)
            test_labels.append(np.array(class_label))

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        if verbose:
            print('train images shape: ', train_images.shape)
            print('train labels shape: ', train_labels.shape)
            print('test images shape: ', test_images.shape)
            print('test labels shape: ', test_labels.shape)
            print(np.unique(train_labels))
        # normalize images
        train_images, test_images = train_images / 255.0, test_images / 255.0

        return (train_images, train_labels), (test_images, test_labels)

    def show_example_images(self, train_images, train_labels):
        '''
        This method shows some example images from [train_images] which is a numpy array as well as [train_labels]
        Uses matplotlib.pyplot
        '''
        plt.figure(figsize=(50,50))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i])

            plt.xlabel(self.class_names[train_labels[i][0]])
        plt.show()


if __name__ == '__main__':
    data_provider = DataProvider()
    data_provider.get_normalized_data(verbose=True)