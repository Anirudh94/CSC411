from PIL import Image
import numpy as np
import glob

def load_data(x_dir, y_csv_file):
	X_train_files = glob.glob(x_dir + '/*.jpg')
	X_train = np.array([np.array(Image.open(fname)) for fname in X_train_files])

	# this is an (N, 1) array; remember to convert to 1-hot for CNN
	y_train = np.genfromtxt(y_csv_file, delimiter=",")[1:,1]
	'''
	print(X_train.shape)
	print(y_train.shape)
	'''
	return X_train, y_train
