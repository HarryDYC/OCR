# ocrgit

[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Optical Character Reader, or OCR for short, is a technology that allows the machine to recognize alphabets on an image and convert them into machine-encoded language. Our version of OCR has implemented machine learning AI so that it can learn from past mistakes and evolve.

## Table of Contents

- [Background](#background)
- [Overview](#overview)
- [Testing](#testing)
- [Files](#files)
- [Future Improvement](#future-improvement)

## Background
OCR is an technology that has been utilized into our everyday life, such as the license plate reader in the parking lot, or the scanner app in our phones. However, the earlier version has detrimental flaws, for example, it cannot record and learn from previous scans to learn from the mistakes. Thanksfully, with AI machine learning technology introduced, we can implement a neural network into OCR, giving it the ability to learn and adapt, therefore becomes more and more accurate.

## Overview
A neural network is a web of inter-connected neurons, with a bias value on each neurons and a weight with each connection. Neurons will form layers, and these layers will form a completed neural network. There are input layer, output layer, and a single or multiple hidden layers. The data will be processed by input layer, forwarded toward the hidden layers for training and calculation, and passed to the output layer for result. 

In the hidden layers, there is a fullt connected layer for training, an activation layer to normalize and de-linearize the data, a loss layer to determine the correctiness of the result, and an accuracy layer to give a percentage presentation of the accuracy. 

To modify the variables to minimize the loss and improve the algorithm, a technology called back propagation is implemented. It will adjust the weights and biases to provide a better model.

## Testing
First testing is conducted with the following perimeters:
* Fully Connect: 26 neurons
* Activation: Sigmoid
* Loss: Quadratic Loss
* Batch Size: 1,000
* Learning Rate: 1,000
* Epoch: 20

The result are:
* Loss: 0.071
* Accuracy: 91.8%

![First Test Loss](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/1_loss.png)
![First Test Accuracy](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/1_accu.png)

This is a pretty good start, but we have to improve more. Since more layers means deeper learning, I add another fully connected layer and activation layer.
Second testing adjusted perimeters:
* Fully Connect: 26 neurons x 2
* Activation: Sigmoid x 2

The results are:
* Loss: 0.281
* Accuracy: 71.2%

![Second Test Loss](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/2_loss.png)
![Second Test Accuracy](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/2_accu.png)

The result is even worse, however, increasing the epoch(letting it trains more rounds) seems to make the result better than having one layer.
Third testing adjusted perimeter:
* epoch: 100

The results are:
* Loss: 0.27
* Accuracy: 97.6%

![Third Test Loss](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/3_loss.png)
![Third Test Accuracy](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/3_accu.png)

Turns out the model is just adapting "slower" than usual. The reason of that is due to the gradient vanishing problem. Due to the nature of the derivative of Sigmoid function, the derivative value(which is essential to adjusting the variables to make the model better) will be close to zero. There are two ways to resolve this problem: change an activation function, or change a loss function.

First step is to change the activation function, since Sigmoid is causing the gradient vanishing problem. Leaky Relu replacing Sigmoid can theoretically solve the issue since it won't have derivative close to zero. 

It's important to note that the maximum derivative value of Sigmoid is 0.25, and that of Leaky Relu is 1. Therefore, the learning rate needs to be adjusted to 1/4, otherwise the variable calibration will oscillate.

Forth testing adjusted perimeters:
* Activation: Leaky Relu
* Learning Rate: 250

The results are:
* Loss: 0.152
* Accuracy: 97.7%

![Forth Test Loss](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/4_loss.png)
![Forth Test Accuracy](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/4_accu.png)

Another method is to change the loss function that will compliment with Sigmoid. Cross Entropy Loss can be used because during back propagation, the denominator of Cross Entropy Loss can eliminate Sigmoid during multiplication, thus negating the gradient vanishing problem.

Same as Leaky Relu, the learning rate should be adjusted accordingly. The previous learning rate of 250 causes the variable calibration to reach negative infinite, suggesting that the learning rate is far too large. To accomodate, I set the learning rate to 2.

Fifth testing adjusted perimeters:
* Activation: Sigmoid
* Loss: Cross Entropy Loss
* Learning Rate: 2

The results are:
* Loss: 0.066
* Accuracy: 99.1%

![Fifth Test Loss](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/5_loss.png)
![Fifth Test Accuracy](https://github.com/HarryDYC/OCR/blob/main/TestResultGraph/5_accu.png)

With this modification, the model reaches its peak performance.

## Files
1. pic
    * Pixel alphabetic images for training and testing.
2. OCR_train.py
    * Used to train the model and present the result.
3. Preprocess.py
    * Used to convert images into numpy array.

## Future Improvement
Although the model can be almost 100% accurate, there are still adjustments to explore and potentially make the model more perfect, such as multiple layers with different activation functions. Also, another form of neural network will also be an effective alternative, such as Comvolutional Neural Network. And of course, everyones suggestions and comments are welcomed!
