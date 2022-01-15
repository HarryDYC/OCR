# encoding=utf-8
from scipy import misc
import imageio as iio
import numpy as np


def main(src, dst):
    with open(src, 'r') as f:
        list = f.readlines()
    data = []
    labels = []
    for i in list:
        name, label = i.strip('\n').split(' ')
        print(name + ' processed')
        img = iio.imread(name)
        img = img/255
        img.resize((img.size, 1))
        data.append(img)
        labels.append(int(label))

    print('write to npy')
    np.save(dst, [data, labels])
    print('completed')


if __name__ == "__main__":
    main('test.txt', 'testtest.npy')
