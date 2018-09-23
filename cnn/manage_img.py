from PIL import Image
import numpy as np
import os
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_image_to_single(img_name):
    im = Image.open(img_name).convert('L')
    table = np.array(im)
    rows, cols = table.shape
    for i in range(rows):
        # print(table[i])
        for j in range(cols):
            if table[i, j] <= 200:
                table[i, j] = 1
            else:
                # table[i, j] = 255
                table[i, j] = 0
    a = table[2:-2,  3:19]
    b = table[2:-2, 20:36]
    c = table[2:-2, 37:53]
    d = table[2:-2, 54:70]
    return a, b, c, d
    # arr = []
    # arr.append(a)
    # arr.append(b)
    # arr.append(c)
    # arr.append(d)
    # return np.array(arr)


def read_name_to_single(img_name):
    data = []
    # print(data)
    for char in img_name:
        # print(char)
        if char.isdigit():
            data.append(ord(char)-ord('0'))
        else:
            data.append(ord(char) - ord('A')+10)
    # print(data)
    # print(to_categorical(3, 8))
    labels_categorical = [to_categorical(label, 36) for label in data]
    # data.append(36)
    # print(labels_categorical)
    # print(labels_categorical.shape)
    return np.array(labels_categorical)


def read_4list(lis):
    # print(data)
    data = ""
    for n in lis:
        # print(char)
        if n < 10:
            data += chr(ord('0')+n)
            # data.append(chr(ord('0')+n))
        else:
            data += chr(ord('A')-10+n)
            # data.append(chr(ord('A')-10+n))
    # print(data)
    return data


def load_data(path):
    images = []
    labels = []
    # if not os.path.isdir(file):
    #     return
    dir_list = os.listdir(path)
    dir_list.sort()
    for dir in dir_list:
        # print(dir)
        for img in os.listdir(path+'/'+dir):
            images_four = read_image_to_single(path+'/'+dir+'/'+img)
            labels_four = read_name_to_single(dir)
            for i in range(4):
                images.append(images_four[i])
                labels.append(labels_four[i])

    print('load success!')
    X = np.array(images)
    # print (X.shape)
    # print(X[0])
    X = X[:, :, :, np.newaxis]
    # X.reshape(11900, 1, 30, 70)
    print(X.shape)
    # print(X[0])

    Y = np.array(labels)
    print(Y.shape)
    # print(Y[0][0])
    return X, Y


def load_one_dir(path):
    img_list = os.listdir(path)
    images = []
    labels = []
    for img in img_list:
        images_four = read_image_to_single(path + '/' + img)
        labels_four = read_name_to_single(img[:4])
        for i in range(4):
            images.append(images_four[i])
            labels.append(labels_four[i])
    print('load success!')
    X = np.array(images)
    X = X[:, :, :, np.newaxis]
    print(X.shape)
    Y = np.array(labels)
    print(Y.shape)
    return X, Y


if __name__ == '__main__':

    # data = read_image_to_single('../IMG/2ATR.jpg')
    # print(data)
    # print(read_name_to_single('2A9Z'))
    # data = np.array(data)
    # # print(data)
    # z = data[2:-2, 3:19]
    # im = Image.fromarray(z)
    # im.save('z.jpg')
    # x = data[2:-2, 20:36]
    # im = Image.fromarray(x)
    # im.save('x.jpg')
    # p = data[2:-2, 37:53]
    # im = Image.fromarray(p)
    # im.save('p.jpg')
    # h = data[2:-2, 54:70]
    # im = Image.fromarray(h)
    # im.save('h.jpg')
    # read_4list([0, 10, 9, 35])
    # read_image('temp.jpg')
    load_data('../TEST')
    # a= read_name_flatten('0A9Z')
    # print(a.shape)
    # print(a)
    # print(read_data(a))
    a = load_one_dir('../IMG')
