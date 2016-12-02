from PIL import Image
import numpy as np
import glob
from sklearn import cluster, datasets
from scipy import misc
import csv

def load_X(x_dir):
	X_train_files = glob.glob(x_dir + '/*.jpg')
	X_train_files.sort()
	X_train = np.array([np.array(Image.open(fname)) for fname in X_train_files])
	
	'''
	X_train = np.array([np.array(Image.open(fname).convert('L')) for fname in X_train_files])
	X_train = X_train[..., np.newaxis]
	'''
	
	return X_train

def load_data(x_dir, y_csv_file):
	X_train = load_X(x_dir)
	
	# this is an (N, 1) array; remember to convert to 1-hot for CNN
	y_train = np.genfromtxt(y_csv_file, delimiter=",")[1:,1]
	print(X_train.shape)
	print(y_train.shape)
	
	for i in range(1,9):	
		print(np.sum(y_train == i))

	return X_train, y_train

def writeCSV(csvfile, pred):
	pred = np.concatenate((pred, np.ones((2000,), dtype=np.int)), axis=0)
	print(pred.shape)
	indices = np.array(range(1,pred.shape[0] + 1))
	pred = np.column_stack((indices,pred,))
	print('shape')
	print(pred.shape)
	with open(csvfile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(pred)

def writeCSV2(csvfile, pred):
	print(pred.shape)
	indices = np.array(range(1+970, pred.shape[0] + 1 + 970))
	pred = np.column_stack((indices,pred,))
	print('shape')
	print(pred.shape)
	with open(csvfile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(pred)


def compressImg(x_dir, size=7000, test=0):
    prefix = ''
    if size == 1:
        prefix = '00001'
    elif size == 10:
        prefix = '0000'
    elif size == 100:
        prefix = '000'
    elif size == 1000:
        prefix = '00'

    if test == 1:
        prefix = 'test_' + prefix

    X_train_files = glob.glob(x_dir + '/' + prefix + '*.jpg')
    X_train_files.sort()

    for fname in X_train_files:
        data = np.array(Image.open(fname))
        data = np.array(data).reshape(-1, 1)

        k_means = cluster.KMeans(n_clusters=3)
        k_means.fit(data)

        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        data_compressed = np.choose(labels, values)
        data_compressed.shape = data.shape
        misc.imsave('compressed/'+ fname.split("/")[1], data_compressed.reshape(128, 128, 3))
