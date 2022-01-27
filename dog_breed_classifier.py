from sklearn.model_selection import learning_curve
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from const import image_size

class DogBreedClassifier():
    def setup_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(image_size[0], (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(2 * image_size[0], (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(image_size[0], (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2))) 

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(2 * image_size[0], activation='relu'))
        self.model.add(layers.Dense(2 * image_size[0], activation='relu'))
        self.model.add(layers.Dense(121))


    def compile_and_train(self, train_images, train_labels, test_images, test_labels):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001, epsilon=1e-07, amsgrad=False,
                name='Adam'
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            train_images, train_labels, 
            epochs=100, 
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
