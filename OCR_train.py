# encoding = utf-8

import imageio as iio
import numpy as np
from scipy import misc

#
# def Preprocess(src):
#     def Dilate(img):
#         kernel = np.ones((3, 3), np.uint8)
#         nimg = misc.dilate(img, kernel, iterations=1)
#         return nimg
#
#     img = iio.imread(src, 0)  # Reads in image in grayscale
#     h, w = img.shape  # height and width
#     f = False
#     while h >= 34 and w >= 34:
#         img = misc.resize(img, None, fx=0.5, fy=0.5, interpolation=misc.INTER_LANCZOS4)
#         h, w = img.shape
#         if f:  # Alternating dilation
#             img = Dilate(img)
#             f = False
#         else:
#             f = True
#         # cv2.imshow("Image",img)
#         # cv2.waitKey(0)
#
#     img = misc.resize(img, (17, 17))
#     img[img > 0] = 1
#     img.resize((289, 1))
#     return img

class Data:
    def __init__(self, name, batch_size):
        with open(name, 'rb') as f:
            data = np.load(f, allow_pickle=True)
        self.x = data[0]
        self.y = data[1]
        self.l = len(self.x)
        self.batch_size = batch_size
        self.pos = 0

    def forward(self):
        batch = self.batch_size
        l = self.l
        if self.pos + batch >= l:
            ret = (self.x[self.pos:l], self.y[self.pos:l])
            self.pos = 0
            index = range(l)
            np.random.shuffle(list(index))
            self.x = self.x[index]
            self.y = self.y[index]
        else:
            ret = (self.x[self.pos:self.pos + batch], self.y[self.pos:self.pos + batch])
            self.pos = self.pos + batch

        return ret, self.pos


class FullyConnect:
    def __init__(self, l_x, l_y):
        self.weight = np.random.randn(l_y, l_x) / np.sqrt(l_x)
        # / np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)

    def forward(self, x):
        self.x = x
        self.y = np.array([np.dot(self.weight, xx) + self.bias for xx in x])
        return self.y
        # x * weight + bias

    def backward(self, d):
        self.dx = np.array([np.dot(self.weight.T, xx) for xx in d])
        cur_dw = self.x
        ddw = np.array([np.dot(dd, xx.T) for dd, xx in zip(d, cur_dw)])
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.weight = self.weight - self.lr * self.dw
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        self.bias = self.bias - self.lr * self.db

        return self.dx


class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2
        return self.loss
        # sum((x - label)^2) / batch / 2

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx


class Sigmoid:
    def __init__(self):
        pass

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # print("forward", x[0])
        self.x = x
        return self.sig(x)

    def backward(self, d):

        sig = self.sig(self.x)
        dx = d * sig * (1 - sig)
        return dx


class RELU:
    def __init__(self):
        pass

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.x = x
        return self.relu(x)

    def backward(self, d):
        dx = d * np.where(self.relu(self.x) > 0, 1, 0)
        # self.relu(self.x)
        return dx


class leakyRELU:
    def __init__(self):
        pass

    def leakyRelu(self, x):
        return np.maximum(0.01 * x, x)

    def forward(self, x):
        self.x = x
        return self.leakyRelu(x)

    def backward(self, d):
        dx = d * np.where(self.leakyRelu(self.x) > 0, 1, 0.01)
        return dx


class Tanh:
    def __init__(self):
        pass

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def forward(self, x):
        self.x = x
        return self.tanh(x)

    def backward(self, d):
        return d * (1 - np.power(self.tanh(self.x), 2))


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 0.99999
        self.loss = np.nan_to_num(-self.label * np.log(x) - ((1 - self.label) * np.log(1 - x)))
        # print("--------------------befopre: ", self.loss)
        self.loss = np.sum(self.loss) / x.shape[0]

        # print("after: ",self.loss)
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / (self.x * (1 - self.x))
        # print(self.dx[0])
        # print("------")
        return self.dx


class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.accuracy = np.sum(np.argmax(xx) == ll for xx, ll in zip(x, label))
        self.accuracy = 1.0 * self.accuracy / x.shape[0]
        return self.accuracy


def main():
    datalayer1 = Data('train_processed.npy', 1000)
    datalayer2 = Data('validate_processed.npy', 10000)
    inner_layers = []
    inner_layers.append(FullyConnect(17 * 17, 26))
    inner_layers.append(Sigmoid())
    # inner_layers.append(FullyConnect(20, 26))
    # inner_layers.append(Sigmoid())
    # inner_layers.append(FullyConnect(200, 26))
    # inner_layers.append(Sigmoid())
    losslayer = CrossEntropyLoss()
    accuracy = Accuracy()

    for layer in inner_layers:
        layer.lr = 2.0

    epochs = 100
    for i in range(epochs):
        print('=========epochs ', i)
        loss_sum = 0
        iter = 0
        while True:
            data, pos = datalayer1.forward()
            x, label = data
            for layer in inner_layers:
                x = layer.forward(x)

            loss = losslayer.forward(x, label)
            loss_sum += loss
            iter += 1
            d = losslayer.backward()

            for layer in inner_layers[::-1]:
                d = layer.backward(d)

            if pos == 0:
                data, _ = datalayer2.forward()
                x, label = data
                for layer in inner_layers:
                    x = layer.forward(x)
                accu = accuracy.forward(x, label)
                print('loss: ', loss_sum / iter)
                print('accuracy: ', accu)
                break
    #
    # while True:
    #     print("Enter path of image:")
    #     path = input()
    #     if (path == "bye" or path == "quit" or path == "exit"):
    #         break
    #     dat = Preprocess(path)
    #     x = []
    #     for i in range(1, 26):
    #         x.append(dat)
    #
    #     for layer in inner_layers:
    #         x = layer.forward(x)
    #
    #     # ASCII A 65 Z 90
    #     print(chr(np.argmax(inner_layers[1].y) + 65))
    #     # os.remove("temp.png")


if __name__ == "__main__":
    main()

#
#     y = a*x + b  if a and b is differnt, then the graph is different
# J(theta ) = h(theta, x) - y
#
# 1000 d(COST) /d(THETA)
#
#
# 1000 cOST --> 1000_COST
# 1000 d(1000_COST) /d(THETA)
#
#     Forward: Data  ----> FullyConnect  --> Sigmoid --> Crossentropy
#
#     Forward: Data  ----> FullyConnect  --> Sigmoid --> QuafraticLost
#                   x1==     x1-> y1   y1 == x2-> y2 ==== x3 ---> y3
#
#     Backward: Data  <---- FullyConnect  <-     - Sigmoid         <-- QuafraticLostt
#                            dx1=d(y3)/d(x1)        d2=d(y2)/d(x2) *  d3  ==   d3 = d(y3)/d(x3)
#                            dw1= d(y3)/d(weight) = d(y3)/d(y1) * d(y1)/d(weight)
#
# d(y2)/d(x2)* d(y3)/d(y2) = d(y3)/d(x2)
#     Get smaller Lost --> Gradient Decent ->> Back Propogation
#
#
# CNN
# Convolutional layer
# max-pooling layer
#
#
#
# fULLY --<
#     1                        2                    8
#
#      e^-1                     e^-2                e^-8
# --------------       +  ---------------    +      -------------
# sum(e^-1  e^-2 ....)     sum(e^-1  e^-2 ....)     sum(e^-1  e^-2 ....)
#
#
#
#
# batch_size  * image_size  --> Input for FC
# 1000       *   289 *1
#
#
#
#
# 1 1 1 1 0 0 0 1 1 1 1 // 289 numbers for image1
# 1 1 1 1 0 0 0 1 0 1 0 // 289 numbers for image2
# ....
# ....
# 1 0 1 1 1 0 0 1 1 0 1 // 289 numbers for image1000
#
# 1 1 0 0 0 1 1 0 0 1 1 // 289 numbers for average
# 1 1 0 0 0 0 0 0 1 0 1 // 289 numbers for variance
#
#
# 187 178 145 198 --> 186 average
# ((187 - 186) ^ 2 + (178 - 186) ^ 2 + (145 - 186) ^ 2 + (198 - 186) ^ 2 ) / 4
#
#
# 174 - 185
# ------
# 5
# Batch Nor --> Take care of the input in order not to depend on the
# assumption that each layer gradeint is independet of the otehr layers!