from dog_breed_classifier import DogBreedClassifier
from data_provider import DataProvider
from parameters_provider import args

if __name__ == '__main__':
    data_provider = DataProvider(args.image_dim)

    (train_images, train_labels), (test_images, test_labels) = data_provider.get_normalized_data(verbose=True)

    # data_provider.show_example_images(train_images, train_labels)

    classifier = DogBreedClassifier(
        args.image_dim,
        args.learning_rate,
        args.epochs,
        args.optimizer,
        args.epsilon,
        args.kernel_size,
        args.dense_neurons
    )

    classifier.setup_model()

    classifier.print_model_summary()

    classifier.compile_and_train(train_images, train_labels, test_images, test_labels)
    
    classifier.evaluate_model(test_images, test_labels)