'''

    Nearesrt Neighbour Implementation
    nearestNeigbour.py

    Created By Ryan Maugin

'''

import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.datasets as datasets


class NearestNeighbour(object):
    ''' Nearest Neighbour Classification implementation class. '''


    def __init__(self):
        ''' Class constructor defining dataset set of images (X) and labels (y). '''

        self.X_train = None
        self.y_train = None


    def train(self, X, y):
        ''' Training classifier simply means copying training data to memeory.

        Parameters:
            - X (list): A list of 50k 30x30 images with 3 colour channels (30x30x3)
            - y (list): Corresponding label for each image in list X.
        '''

        self.X_train = X
        self.y_train = y


    def predict(self, input_image, input_label):
        ''' Perform classification (prediction) on given "test_input" image.

        Parameters:
            - input_image (tensor): A 30x30 image with 3 colour channels (30x30x3)
            - input_label (int): The correct corresponding label for the image.

        Returns:
            Most similar image and classficiation label predicted for input.
        '''

        # Stores image, label and summed difference of image with smallest L1
        # distance from input image (img, label, difference)
        closest_img = None

        # Get the size of the trainset
        size_of_trainset = np.shape(self.X_train)[0]

        # Loop through each image in the training set
        for img in range(0, size_of_trainset):

            # Calculate the pixelwise difference of input img and img from dataset
            # then get the absoulte sum of resultant tensor
            difference = np.abs(np.sum(np.subtract(input_image, self.X_train[img])))

            # Check if this is first comparison init closest_img to first img
            if closest_img == None:
                closest_img = (self.X_train[img], self.y_train[img], difference)
                continue

            # if current img difference is less than current closest_img value replace it
            if difference < closest_img[2]:
                closest_img = (self.X_train[img], self.y_train[img], difference)

        return closest_img


if __name__ == '__main__':

    # Create an instance of the nearest neigbout classifier
    classifier = NearestNeighbour()

    # All the classes in index order that can be classified
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

    # Download CIFAR10 benchmark dataset and put it in ./data folder
    cifar_10_trainset = datasets.CIFAR10(root='./data', train=True, download=True)

    # Define training data and labels to be passed in to train method
    training_data     = cifar_10_trainset.train_data
    training_labels   = cifar_10_trainset.train_labels

    # Nearest Neigbour class training method which takes training data (X, y)
    classifier.train(training_data, training_labels)

    # Downlaod the CIFAR10 benchmark test set and put it in ./data folder
    cifar_10_testset  = datasets.CIFAR10(root='./data', train=False, download=True)

    # Define test data and get only one random test example to predict
    random_example    = random.randint(0, 9999)
    testing_data      = cifar_10_testset.test_data[random_example]
    testing_label     = cifar_10_testset.test_labels[random_example]

    # Call the prediction method which will return nearest neigbour for input
    prediction = classifier.predict(testing_data, testing_label)

    # Plot the results to graphically illustrate results
    f, img_plots = plt.subplots(1,2)
    img_plots[0].imshow(testing_data)
    img_plots[0].set_title("Input: {}".format(classes[testing_label]))
    img_plots[1].imshow(prediction[0])
    img_plots[1].set_title("Output: {}".format(classes[prediction[1]]))
    plt.xlabel("Pixel Difference: {}".format(prediction[2]))
    plt.show()
