from sklearn.model_selection import learning_curve
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from const import image_size

class DogBreedClassifier():

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
        self.model.summary()

    def evaluate_model(self, test_images, test_labels):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)

        return (test_loss, test_acc)
