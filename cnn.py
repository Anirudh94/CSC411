from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

import matplotlib.pyplot as plt

from util import load_data

# dimensions of our images.
img_width, img_height, img_channels = 128, 128, 3
nb_classes = 8
nb_epoch = 50
batch_size = 32

# load training data
X_train, y_train = load_data('./train', 'train.csv')

X_train = X_train[:6000]
y_train = y_train[:6000]

X_val = X_train[6000:6800]
y_val = y_train[6000:6800]

X_test = X_train[6800:]
y_test = y_train[6800:]

# convert data to one-hot
y_train = to_categorical(y_train - 1, nb_classes=nb_classes)
y_val = to_categorical(y_val - 1, nb_classes=nb_classes)
y_test = to_categorical(y_test - 1, nb_classes=nb_classes)

# build ConvNet
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, img_channels)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, img_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
          nb_epoch=nb_epoch,
          batch_size=batch_size,
          validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, batch_size=batch_size)

print('score')
print(score)

model.save_weights('weights.h5')

# plot learning curves
plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='valid accuracy')
plt.legend()
plt.show()
 
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.legend()
plt.show()
