from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from util import load_data
from keras.utils.np_utils import to_categorical

# dimensions of our images.
img_width, img_height, img_channels = 128, 128, 3
nb_train_samples = 7000
nb_classes = 8
nb_epoch = 50
batch_size = 32

# load training data
X_train, y_train = load_data('./train', 'train.csv')

# convert data to one-hot
y_train = to_categorical(y_train - 1, nb_classes=nb_classes) 

# build ConvNet
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, img_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          nb_epoch=nb_epoch,
          batch_size=batch_size)

score = model.evaluate(X_train, y_train, batch_size=batch_size)

print('score')
print(score)

model.save_weights('weights.h5')
