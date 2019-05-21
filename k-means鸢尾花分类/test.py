import struct
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from PIL import Image
def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head


if __name__ == "__main__":
    file1 = 'Mnist/train-images-idx3-ubyte/train-images.idx3-ubyte'
    file2 = 'Mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte'
    file3 = 'Mnist/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
    file4 = 'Mnist/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'

    imgs_train, data_head_train = loadImageSet(file1)
    imgs_test, data_head_test = loadImageSet(file4)
    labels_train, labels_head_train = loadLabelSet(file2)
    labels_test, labels_head_test = loadLabelSet(file3)

    origin_7_imgs = []
    for i in range(1000):
        if labels_train[i] == 7 and len(origin_7_imgs) < 100:
            origin_7_imgs.append(imgs_train[i])


    def array_to_img(array):
        array = array * 255
        new_img = Image.fromarray(array.astype(int))
        return new_img


    def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
        new_img = Image.new(new_type, (col * each_width, row * each_height))
        for i in range(len(origin_imgs)):
            each_img = array_to_img(np.array(origin_imgs[i]).reshape(each_width,each_width))
            new_img.paste(each_img, ((i % col) * each_width, (i / col) * each_width))
            return new_img


    ten_origin_7_imgs = comb_imgs(origin_7_imgs, 10, 10, 28, 28, 'L')
    ten_origin_7_imgs.show()