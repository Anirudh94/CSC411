from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.constraints import maxnorm
from keras import backend as K
K.set_image_dim_ordering('tf')

from util import load_data

# dimensions of our images.
img_width, img_height, img_channels = 128, 128, 3
nb_classes = 8
nb_epoch = 25
batch_size = 32

# load training data
X, y = load_data('./train', 'train.csv')

X_train = X
y_train = y

X_val = X[6500:]
y_val = y[6500:]

# normalize inputs from 0-255 to 0.0-1.0
'''
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train / 255.0
X_val = X_val / 255.0
'''

# convert target data to one-hot
y_train = to_categorical(y_train - 1, nb_classes=nb_classes)
y_val = to_categorical(y_val - 1, nb_classes=nb_classes)

# build ConvNet
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, img_channels), border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
lrate = 0.01
decay = float(lrate)/nb_epoch;
sgd = SGD(lr=lrate, decay=decay, momentum=0.2, nesterov=False)
model.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train,
	batch_size=batch_size,
    nb_epoch=nb_epoch)

'''
# this will do preprocessing and realtime data augmentation
train_datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_datagen.fit(X_train)

model.fit_generator(train_datagen.flow(X_train, y_train,
	batch_size=batch_size),
    samples_per_epoch=X_train.shape[0],
    nb_epoch=nb_epoch,
	validation_data=(X_val, y_val))
'''
model.save_weights('weights.h5')
