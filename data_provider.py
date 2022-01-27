import os
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from const import class_names
from PIL import Image
import numpy as np
import random
from const import data_path, image_size

class DataProvider:
    def get_cifar_data(self, verbose=False):
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0

        if verbose:
            print('train images shape: ', train_images.shape)
            print('train labels shape: ', train_labels.shape)
            print('test images shape: ', test_images.shape)
            print('test labels shape: ', test_labels.shape)

            print(train_labels[0:10])

        return (train_images, train_labels), (test_images, test_labels)

    def show_example_images(self, train_images, train_labels):
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i])

            plt.xlabel(self.class_names[train_labels[i][0]])
        plt.show()

    def get_normalized_data(self, verbose=False):
        self.class_names = []
        class_names_dict = []
        data = []

        print('Iterating through data...')
        i = 0
        for class_name in os.listdir(data_path):
            class_names_dict.append({class_name[10:]: i})
            self.class_names.append(class_name[10:])
            i += 1
            if os.path.isdir(f'{data_path}/{class_name}'):
                for image in os.listdir(f'{data_path}/{class_name}'):
                    
                    image_array = np.asarray(Image.open(f'{data_path}/{class_name}/{image}').resize(image_size))

                    data.append((image_array, i))

        random.shuffle(data)
        
        print('Got data!')
        
        train_size = int(0.8 * len(data))
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

        if verbose:
            # we need to convert this shit to numpy arrays
            # and have the same shape as in cifar data
            print('train images shape: ', train_images.shape)
            print('train labels shape: ', train_labels.shape)
            print('test images shape: ', test_images.shape)
            print('test labels shape: ', test_labels.shape)
        
        # normalize images
        # train_images, test_images = train_images / 255.0, test_images / 255.0

        return (train_images, train_labels, test_images, test_labels)

data_provider = DataProvider()
data_provider.get_cifar_data(verbose=True)
data_provider.get_normalized_data(verbose=False)