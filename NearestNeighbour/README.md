


```

Nearest Neighbour Implementation
===============================

In this section, I implement a trivial and impractical classifier called Nearest
Neighbour which will take in an image as an input and will spit out a similar image
and a classification label as an output.


What is Nearest Neighbour Classifier?
----------------------------------------------------
This trivial and simple classifier which begins by taking an image tensor as an
input with dimensions of 30x30x3 which signifies size of image as 30x30 and includes
3 colour channels (RGB). The training process simply copies test data inputs and
labels (X, y) into memory.

Secondly, the prediction process is a simple algorithm which will get the summed
up absolute tensor pixel-wise difference for the input image and an image in the
training set (repeated for all examples in training set).

Finally, we store all of the results for each example and output the example that
 has the smallest difference to the input image.

This is known as the L1 or Manhattan distance algorithm.


Why is this classifier not used?
----------------------------------------------------
This classifier is not used for many reason, primarily being that it will only output
the label of the image that is similar in terms of the absolute tensor difference
for each pixel in the given image tensor and all other image tensor in the dataset.
As a result a picture of a boat in the blue ocean might very likely match to a
picture of a plane in the blue sky, which is not what we want.

This classifier also has very fast constant training time of O(1) and slow prediction
time of O(n) which can be very slow if 'n' is large (which depends on your benchmark
dataset, mine being CIFAR-1O). This is the opposite of what we usually want as the
prediction will usually run in production devices and should be as efficient as possible.


Want to try my implementation out?
----------------------------------------------------
I have made it so that the output of running this classifier is a graphical plot
of the input and output image along with their respective label, accompanied with the summed up absolute pixel-wise difference of both images.

1) Clone the implementation: git clone https://github.com/RyanMaugin/MachineLearningRevisionBook/tree/master/NearestNeighbour
2) Navigate to project folder in your terminal.
3) Execute the command: "python nearestNeigbour.py"

Note: First execution may take a while as it will need to download CIFAR-10 dataset which is ~200MB big.


Motivation of implementation?
----------------------------------------------------
I implemented this classifier as optional homework to the "Stanford CS231n Convolutional
Neural Networks for Visual Recognition" online course I am following to learn the theory,
implementation and practicality of CNN's.


Author
----------------------------------------------------
🤓 Ryan Maugin
🐦 @techedryan
📧 ryanmaugin@icloud.com
```
