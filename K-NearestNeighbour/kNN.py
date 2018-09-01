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


    def __init__(self, k, metric):
        ''' Class constructor defining dataset set of images (X) and labels (y).

        Parameters:
            - k (int): Number of nearest neigbours we want to compare to.
            - metric (str): "L1" or "L2" distance algorithm to use.
        '''

        self.k       = k
        self.metric  = metric
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


    def majority_vote(self, closest_imgs):
        ''' Outputs the majority label from k nearest neigbours.

        Parameters:
            - closest_imgs (list): Input image k nearest neighbours.

        Returns label which appears the most (majority vote).
        '''

        # Holds appearence count for each label
        label_count = {}

        # For each neighbour (img) in closest_imgs list ...
        for neighbour in closest_imgs:
            # ... if it's first time appearence of label ...
            if neighbour[1] not in label_count:
                # ... add it to label_count with label as key and value as 1.
                label_count[neighbour[1]] = 1
            else:
                # ... else increment current label count by one
                label_count[neighbour[1]] += 1


        # Gets the most biggest number of occurence from sorted
        most_occurent_lbl = max(label_count.values())
        # Holds the label(s) with max count (most occurences)
        majority_labels   = []

        # For each element in label_count dictionary ...
        for key in label_count:
            # ... if label key we are at is one of the most occurent label then ...
            if label_count[key] == most_occurent_lbl:
                # ... append it to the majority_labels list
                majority_labels.append(key)

        # When theres more than one majority label just pick first one in list
        return majority_labels[0]


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
        closest_imgs = []

        # Get the size of the trainset
        size_of_trainset = np.shape(self.X_train)[0]

        # Loop through each image in the training set
        for img in range(0, size_of_trainset):

            # Check if L1 (Manhattan) distance algorithm was chosen as metric
            if self.metric == "L1":
                # Calculate difference using L1 distance algorithm
                difference = np.sum(np.abs(np.subtract(input_image, self.X_train[img])))
            elif self.metric == "L2":
                img_difference = np.square(np.subtract(input_image, self.X_train[img]))
                difference     = np.sqrt(np.sum(img_difference))

            # Append tuple of (input_image, predicted_label, difference)
            closest_imgs.append((self.X_train[img], self.y_train[img], difference))

            # If there is 100 elements in closest_imgs list then sort the tuples
            # from smallest difference to biggest for input img and keep the k smallest items
            if len(closest_imgs) == 100 or img == (size_of_trainset - 1):
                closest_imgs.sort(key=lambda tup: tup[2])
                del closest_imgs[self.k:len(closest_imgs)]

        # Perform majority vote on k examples with smallest distance from input image
        predicted_label = self.majority_vote(closest_imgs)

        returned_example = None
        for x in closest_imgs:
            if x[1] == predicted_label:
                returned_example = x
                break

        return (predicted_label, returned_example)


if __name__ == '__main__':

    # Create an instance of the nearest neigbout classifier
    classifier = NearestNeighbour(7, "L2")

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

    print(prediction[1])
    print("{}: {}".format(prediction[0], classes[prediction[0]]))

    # Plot the results to graphically illustrate results
    f, img_plots = plt.subplots(1,2)
    img_plots[0].imshow(testing_data)
    img_plots[0].set_title("Input: {}".format(classes[testing_label]))
    img_plots[1].imshow(prediction[1][0])
    img_plots[1].set_title("Output: {}".format(classes[prediction[0]]))
    plt.xlabel("Pixel Difference: {}".format(prediction[1][2]))
    plt.show()
