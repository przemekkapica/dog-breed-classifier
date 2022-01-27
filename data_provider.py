import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from const import class_names

class DataProvider:
    def get_normalized_data(self, verbose=False):
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0

        if verbose:
            print('train images shape: ', train_images.shape)
            print('train labels shape: ', train_labels.shape)
            print('test images shape: ', test_images.shape)
            print('test labels shape: ', test_labels.shape)

        return (train_images, train_labels), (test_images, test_labels)

    def show_example_images(self, train_images, train_labels):
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i])
            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            plt.xlabel(class_names[train_labels[i][0]])
        plt.show()