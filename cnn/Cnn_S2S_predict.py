import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, TimeDistributed, GRU
from keras.layers.core import Dense, RepeatVector, Activation
from keras.models import Sequential
import load_img
import urllib
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

input_shape = (30, 70, 1)

model = Sequential()

model.add(Conv2D(100, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(200, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(RepeatVector(1))

model.add(s2s_model)
model.add(TimeDistributed(Dense(36)))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr = 0.1),
              metrics=['accuracy'])
model.load_weights('model_weights/0.146.h5', by_name=True)
src = 'http://210.39.12.30/xsxkapp/sys/xsxkapp/student/4/vcode.do?timestamp=1537496995081'

for i in range(1):
    res = requests.get(src)
    token = res.json()['data']['token']
    # print(token)
    imgurl = 'http://210.39.12.30/xsxkapp/sys/xsxkapp/student/vcode/image.do?vtoken=' + token
    rr = []
    for i in range(100):
        urllib.urlretrieve(imgurl, 'temp.jpg')
        imgplot = plt.imshow(mpimg.imread('temp.jpg'))
        x = np.array([load_img.read_image('temp.jpg')])
        r = model.predict(x)
        result = np.argmax(r, 2)
        rr.append(result[0])
    plt.ion()
    plt.show()
    rr = np.array(rr)
    # print(rr)
    rr = np.transpose(np.array(rr))
    # print(rr)
    count = []
    for row in rr:
        count.append(np.argmax(np.bincount(row)))
    # print(count)
    captcha = load_img.read_4list(count)
    print(captcha)


    # x = np.array([load_img.read_image('TRAIN_1000/2H2R/5.jpg')])
    # r = model.predict(x)
    # result = np.argmax(r, 2)
    # print(load_img.read_4list(result[0]))
    # exit()