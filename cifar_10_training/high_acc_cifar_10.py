import keras
import tensorflow

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers, optimizers

import numpy as np
from matplotlib import pyplot


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print('x_train =', x_train.shape)
print('x_valid =', x_valid.shape)
print('x_test =', x_test.shape)

mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))

x_train = (x_train - mean) / (std + 1e-7)
x_valid = (x_valid - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)

num_classes = 10

y_train = np_utils.to_categorical(y_train, num_classes)
y_valid = np_utils.to_categorical(y_valid, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )

datagen.fit(x_train)


# base_hidden_units - number of hidden units variable
# weight_decay - L2 regularization hyperparameter
def get_model(base_hidden_units=32, weight_decay=1e-4):
    md = Sequential()

    # CONV_1
    md.add(Conv2D(base_hidden_units, kernel_size=3, padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay),
                  input_shape=x_train.shape[1:]))
    md.add(Activation('relu'))
    md.add(BatchNormalization())

    # CONV_2
    md.add(Conv2D(base_hidden_units, kernel_size=3, padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)))
    md.add(Activation('relu'))
    md.add(BatchNormalization())

    # POOL + Dropout
    md.add(MaxPooling2D(pool_size=(2, 2)))
    md.add(Dropout(0.2))

    # CONV_3
    md.add(Conv2D(base_hidden_units * 2, kernel_size=3, padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)))
    md.add(Activation('relu'))
    md.add(BatchNormalization())

    # CONV_4
    md.add(Conv2D(base_hidden_units * 2, kernel_size=3, padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)))
    md.add(Activation('relu'))
    md.add(BatchNormalization())

    # POOL + Dropout
    md.add(MaxPooling2D(pool_size=(2, 2)))
    md.add(Dropout(0.3))

    # CONV_5
    md.add(Conv2D(base_hidden_units * 4, kernel_size=3, padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)))
    md.add(Activation('relu'))
    md.add(BatchNormalization())

    # CONV_6
    md.add(Conv2D(base_hidden_units * 4, kernel_size=3, padding='same',
                  kernel_regularizer=regularizers.l2(weight_decay)))
    md.add(Activation('relu'))
    md.add(BatchNormalization())

    # POOL + Dropout
    md.add(MaxPooling2D(pool_size=(2, 2)))
    md.add(Dropout(0.4))

    # FC_7
    md.add(Flatten())
    md.add(Dense(10, activation='softmax'))

    return md


model = get_model()
model.summary()

batch_size = 128
epochs = 125
model_file_path = 'high_acc_mod_cifar10.hdf5'

checkpointer = ModelCheckpoint(filepath=model_file_path, verpose=1,
                               save_best_only=True)

optimizer = tensorflow.optimizers.Adam(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(datagen.flow(x_train, y_train,
                                           batch_size=batch_size),
                              callbacks=[checkpointer],
                              steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,
                              verbose=2, validation_data=(x_valid, y_valid))

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('Test result: %.3f loss: %.3f' % (scores[1] * 100, scores [0]))

pyplot.plot(history.history['accuracy'], label='train acc')
pyplot.plot(history.history['val_accuracy'], label='test acc')
pyplot.legend()
pyplot.show()

pyplot.plot(history.history['loss'], label='train loss')
pyplot.plot(history.history['val_loss'], label='test loss')
pyplot.legend()
pyplot.show()
