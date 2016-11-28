from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from util import load_data

# dimensions of our images.
img_width, img_height, img_channels = 128, 128, 3
nb_classes = 8
nb_epoch = 100
batch_size = 32
data_augmentation = True
#data_augmentation = False

# load training data
X, y = load_data('./train', 'train.csv')

X_train = X
y_train = y

X_val = X[6000:]
y_val = y[6000:]

# convert data to one-hot
y_train = to_categorical(y_train - 1, nb_classes=nb_classes)
y_val = to_categorical(y_val - 1, nb_classes=nb_classes)

# build ConvNet
model = Sequential()

model.add(Convolution2D(32, 3, 3,
    input_shape=(img_width, img_height, img_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.3, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

'''
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255
'''

if not data_augmentation:
    print('NOT augmenting data')
    model.fit(X_train, y_train,
        nb_epoch=nb_epoch,
        batch_size=batch_size) #, callbacks=[tb])
else:
    print('augmenting data')
    # this will do preprocessing and realtime data augmentation
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)  # randomly flip images

    train_datagen.fit(X_train)

    model.fit_generator(train_datagen.flow(X_train, y_train,
        batch_size=batch_size),
        samples_per_epoch=X_train.shape[0],
        nb_epoch=nb_epoch)

model.save_weights('weights.h5')
# evaluate score
score = model.evaluate(X_val, y_val, batch_size=batch_size)
print('score')
print(score)

