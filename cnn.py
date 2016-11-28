from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from util import load_data

X_train, y_train = load_data('./train', 'train.csv')

# dimensions of our images.
img_width, img_height, img_channels = 128, 128, 3
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50

'''

img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image

#build convnet
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_channels, img_width, img_height)))
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
'''