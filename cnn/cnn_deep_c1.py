import os
import keras
import load_img
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.core import Dense
from keras.models import Sequential


os.environ['CUDA_VISIBLE_DIVICE'] = '1'
x_train, y_train = load_img.load_data_flatten('../TRAIN_1000')
x_vali, y_vali = load_img.load_one_dir_flatten('../IMG')
x_test, y_test = load_img.load_data_flatten('../TRAIN')

print(x_train[0])
print(y_train[0])
input_shape = (30, 70, 1)

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
model.add(Dense(144, activation='relu'))

model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.Adam(lr=0.001),
              optimizer=keras.optimizers.Adadelta(),
              # optimizer=(keras.optimizers.SGD(lr=0.01)),
              metrics=['accuracy'])

batch_size = 32
epochs = 500

filepath = "val_weights/c1-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, validation_data=(x_vali, y_vali),
          batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
          verbose=2)

train_result = model.evaluate(x_train, y_train)
test_result = model.evaluate(x_test, y_test)

print('Train = ', train_result)
print('Test = ', test_result)

model.save('model_weights/' + str(round(test_result[1], 3)) + '.h5')
json_string = model.to_json()
open('model_json/' + str(round(test_result[1], 3)) + '.json', 'w').write(json_string)

print('\nSuccessfully saved as model')

