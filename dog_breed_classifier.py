from sklearn.model_selection import learning_curve
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

class DogBreedClassifier():
    def setup_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))


    def compile_and_train(self, train_images, train_labels, test_images, test_labels):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam'
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            train_images, train_labels, 
            epochs=1, 
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
