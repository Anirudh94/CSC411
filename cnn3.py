import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json

from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from util import *
from sklearn.utils import shuffle

from keras import backend as K
K.set_image_dim_ordering('th')

# path to the model weights file.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 6000
nb_validation_samples = 1000
nb_epoch = 20
nb_classes = 8

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# load training data
X, y = load_data('./train', 'train.csv')
X, y = shuffle(X, y, random_state=0)

X = np.transpose(X, (0, 3, 1, 2))

X_train = X[:nb_train_samples]
y_train = y[:nb_train_samples]
X_test = X[nb_train_samples:]
y_test = y[nb_train_samples:]

# convert data to one-hot
y_train = to_categorical(y_train - 1, nb_classes=nb_classes)
y_test = to_categorical(y_test - 1, nb_classes=nb_classes)

train_generator = train_datagen.flow(X_train, y_train,
		batch_size=32)

validation_generator = test_datagen.flow(X_test, y_test,
		batch_size=32)

# gen submission
'''
X_pub = load_X('./val')
X_priv = load_X('./private')

X_val = np.concatenate([X_pub, X_priv])
X_val = np.transpose(X_val, (0, 3, 1, 2))

pred_onehot = model.predict(X_val, batch_size=32, verbose=1)
print pred_onehot[:10]
pred = np.argmax(pred_onehot, axis=1) + 1
print 'writing to csv file...'
id = np.arange(1, 2970+1, 1)
id_str = id.tolist()
pred = pred.astype(int)
pred_str = pred.tolist()
id_str.insert(0, 'Id') 
pred_str.insert(0, 'Prediction')
vals = list(zip(id_str, pred_str))

print vals[:10]
np.savetxt('submission.csv', vals, delimiter=",", fmt="%s") 
'''

'''
X_val = load_X('./val')
X_val = np.transpose(X_val, (0, 3, 1, 2))
X_priv = load_X('./private')
X_priv = np.transpose(X_priv, (0, 3, 1, 2))

pred = model.predict(X_val, batch_size=32, verbose=1)
pred = np.argmax(pred, axis=1) + 1

writeCSV('submission1.csv', pred)

pred = model.predict(X_priv, batch_size=32, verbose=1)
pred = np.argmax(pred, axis=1) + 1

writeCSV2('submission2.csv', pred)
'''
model.save('my_model.h5') 

'''
y_test = np.argmax(y_test, axis=1) + 1

for i in range(0, 1000):
    print(pred[i],y_test[i])

print(np.sum(pred == y_test))
'''

# fine-tune the model
'''
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
'''

