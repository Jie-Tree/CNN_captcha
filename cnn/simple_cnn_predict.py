
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.core import Dense
from keras.models import Sequential
import manage_img
import urllib.request
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

input_shape = (26, 16, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
# print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# print(model.output_shape)
model.add(Dropout(0.25))
# print(model.output_shape)

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# print(model.output_shape)
model.add(Dropout(0.25))
# print(model.output_shape)

model.add(Conv2D(128, (3, 3), activation='relu'))
# print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
# print(model.output_shape)
model.add(Dropout(0.25))
# print(model.output_shape)

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.Adam(lr=0.001),
              optimizer=keras.optimizers.Adadelta(),
              # optimizer=(keras.optimizers.SGD(lr=0.01)),
              metrics=['accuracy'])

# model.load_weights('val_weights/c0-01-0.17.h5', by_name=True)
model.load_weights('model_weights/0.972.h5', by_name=True)
src = 'http://210.39.12.30/xsxkapp/sys/xsxkapp/student/4/vcode.do?timestamp=1537496995081'

for i in range(1):
    res = requests.get(src)
    token = res.json()['data']['token']
    # print(token)
    imgurl = 'http://210.39.12.30/xsxkapp/sys/xsxkapp/student/vcode/image.do?vtoken=' + token
    rr = []
    for i in range(1):
        # acc enough
        urllib.request.urlretrieve(imgurl, 'temp.jpg')
        imgplot = plt.imshow(mpimg.imread('temp.jpg'))
        x = np.array(manage_img.read_image_to_single('temp.jpg'))
        # print(x.shape)
        x = x[:, :, :, np.newaxis]
        r = model.predict(x)
        # print(r)
        result = np.argmax(r, 1)
        # print(result)
        # print(manage_img.read_4list(result))
        rr.append(result)
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
    captcha = manage_img.read_4list(count)
    print(captcha)

