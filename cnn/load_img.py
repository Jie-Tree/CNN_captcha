import os
from PIL import Image
import numpy as np
from keras.utils.np_utils import to_categorical


def read_data(data):
    # print(data)
    text = str()
    for i in range(4):
        # print(char)
        if 10 <= data[i]:
            text = text+chr(int(data[i])+ord('A')-10)
        else:
            text = text+chr(int(data[i])+ord('0'))
    return text


def read_image(img_name):
    im = Image.open(img_name).convert('L')
    # im = Image.open(img_name).convert('1')
    # im.show()
    data = np.array(im)
    # print(data)
    # print(data.shape)
    data = np.reshape(data,[30,70,1])
    # data = data[:, :, np.newaxis]
    # print(data.shape)
    # print(data[0][69])
    # exit()
    return data


def read_4list(lis):
    data = []
    # print(data)
    for n in lis:
        # print(char)
        if n < 10:
            data.append(chr(ord('0')+n))
        else:
            data.append(chr(ord('A')-10+n))
    # print(data)
    return data


def read_name(img_name):
    data = []
    # print(data)
    for char in img_name:
        # print(char)
        if char.isdigit():
            data.append(ord(char)-ord('0'))
        else:
            data.append(ord(char) - ord('A')+10)
    print(data)
    labels_categorical = [to_categorical(label, 36)[0] for label in data]
    # data.append(36)
    # print(labels_categorical)
    # print(labels_categorical.shape)
    return np.array(labels_categorical)


def read_name_flatten(img_name):
    data = []
    # print(data)
    for char in img_name:
        # print(char)
        if char.isdigit():
            data.append(ord(char)-ord('0'))
        else:
            data.append(ord(char) - ord('A')+10)
    # print(data)
    labels_categorical = np.array([to_categorical(label, 36) for label in data])
    # data.append(36)
    # print(labels_categorical)
    # print(labels_categorical.shape)
    return labels_categorical.flatten()


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
            labels.append(read_name(dir))
            # print(img)
            images.append(read_image(path+'/'+dir+'/'+img))
    print('load success!')
    X = np.array(images)
    # X.reshape(11900, 1, 30, 70)
    print(X.shape)
    Y = np.array(labels)
    print(Y.shape)
    # print(Y[0][0])
    return X, Y


def load_data_flatten(path):
    images = []
    labels = []
    # if not os.path.isdir(file):
    #     return
    dir_list = os.listdir(path)
    dir_list.sort()
    for dir in dir_list:
        # print(dir)
        for img in os.listdir(path+'/'+dir):
            labels.append(read_name_flatten(dir))
            # print(img)
            images.append(read_image(path+'/'+dir+'/'+img))
    print('load success!')
    X = np.array(images)
    # X.reshape(11900, 1, 30, 70)
    print(X.shape)
    Y = np.array(labels)
    print(Y.shape)
    # print(Y[0][0])
    return X, Y


def load_one_dir(path):
    img_list = os.listdir(path)
    images = []
    labels = []
    for img in img_list:
        labels.append(read_name(img[:4]))
        # print(img[:4])
        # print(img)
        images.append(read_image(path + '/' + img))
        # print(img)
    # print(labels)
    print('load success!')
    X = np.array(images)
    # X.dtype=np.uint8
    # X.reshape(11900, 1, 30, 70)
    # print(X[0])
    print(X.shape)
    Y = np.array(labels)
    # Y.dtype=np.uint8
    # print(Y[0])
    print(Y.shape)
    return X, Y


def load_one_dir_flatten(path):
    img_list = os.listdir(path)
    images = []
    labels = []
    for img in img_list:
        labels.append(read_name_flatten(img[:4]))
        # print(img[:4])
        # print(img)
        images.append(read_image(path + '/' + img))
        # print(img)
    # print(labels)
    print('load success!')
    X = np.array(images)
    # X.dtype=np.uint8
    # X.reshape(11900, 1, 30, 70)
    # print(X[0])
    print(X.shape)
    Y = np.array(labels)
    # Y.dtype=np.uint8
    # print(Y[0])
    print(Y.shape)
    return X, Y


if __name__ == '__main__':
    # read_4list([0, 10, 9, 35])
    # read_image('TRAIN/VCUK/0.jpg')
    # load_data('TRAIN')
    a= read_name_flatten('0A9Z')
    print(a.shape)
    # print(a)
    # print(read_data(a))
    # a = load_one_dir('IMG')
    # print(a.shape)