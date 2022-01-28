from sklearn.model_selection import learning_curve
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from const import image_size

class DogBreedClassifier():
    '''
    DogBreedClassifier is a CNN classification model trained to recognize provided dog breeds.
    Takes following parameters:
    [image_dim] - (int) dimension of a image feeded to the model (size of a image is [image_dim]x[image_dim])
    [learning_rate] - (float) determines the step size at each iteration while moving toward a minimum of a loss function
    [epochs] - (int) number of cycles for training
    [optimizer] - (str) algorithm or method used to change the attributes of the model
    [epsilon] - (float) small constant for numerical stability
    [kernel_size] - (int) dimension of a kernel which is a filter that is used to extract the features from the images
    [dense_neurons] - (int) number of neurons used in dense layers of a CNN
    '''

    def __init__(self, image_dim, learning_rate, epochs, optimizer, epsilon, kernel_size, dense_neurons):
        super().__init__()
        self.image_dim = image_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.kernel_size = (kernel_size, kernel_size)
        self.dense_neurons = dense_neurons

    def setup_model(self):
        '''
        Initializes CNN model with provided parameters. 
        Firstly creates a convolutional base with 3 patterns: Conv2D followed by MaxPooling2D layers.
        Then it flattens the output of the last MaxPooling2D layer and feeds it to dense layer (converts 3D to 1D). 
        Dense layers are responsible for actual class recognition. The last dense layer has 121 neurons because 
        there are 121 classes of dog breeds in our dataset.
        '''
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(self.image_dim, self.kernel_size, activation='relu', input_shape=(self.image_dim, self.image_dim, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(2 * self.image_dim, self.kernel_size, activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(self.image_dim, self.kernel_size, activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2))) 

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.dense_neurons, activation='relu'))
        self.model.add(layers.Dense(self.dense_neurons, activation='relu'))
        self.model.add(layers.Dense(121))


    def compile_and_train(self, train_images, train_labels, test_images, test_labels):
        '''
        This method compiles and trains the CNN model with provided parameters.
        [train_images] - (np.array) array with train images pixeled data. Shape - (img_count, img_dim, img_dim, 3)
        [train_labels] - (np.array) array with train labels (classes: 0-120). Shape - (label_count, )
        [test_images] - (np.array) array with test images pixeled data. Shape - (img_count, img_dim, img_dim, 3)
        [test_labels] - (np.array) array with test labels (classes: 0-120). Shape - (label_count, )

        Other parameters are described in DogBreedClassifier docs.
        '''
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, epsilon=self.epsilon, amsgrad=False,
                name=self.optimizer
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            train_images, train_labels, 
            epochs=self.epochs, 
            validation_data=(test_images, test_labels)
        )

    def print_model_summary(self):
        '''
        Prints a useful summary of the model, which includes: Nnme and type of all layers in the model.
        Also outputs shape for each layer.
        '''
        self.model.summary()

    def evaluate_model(self, test_images, test_labels):
        '''
        This method tests how the classifier handles recognizing data it hasn't seen before.
        It takes [test_images] and [test_labels] parameters.
        [test_images] - (np.array) array with test images pixeled data. Shape - (img_count, img_dim, img_dim, 3)
        [test_labels] - (np.array) array with test labels (classes: 0-120). Shape - (label_count, )

        Uses matplotlib.pyplot for plotting accuracy and validation accuracy.

        Returns obtained loss and accuracy for provided data.
        '''
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)

        return (test_loss, test_acc)
