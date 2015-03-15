import pandas as pd
import numpy as np
import pickle
from lasagne import layers
from lasagne.nonlinearities import softmax, tanh
from lasagne.updates import nesterov_momentum, sgd
from sklearn.preprocessing import MinMaxScaler
from nolearn.lasagne import NeuralNet
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

## data input 

trans = MinMaxScaler()

data = pd.read_csv('training.csv', header = 0).dropna()

y = np.array(data[[col for col in data if col != 'Image']]).astype(dtype = np.float32)
X = ' '.join([x for x in np.array(data['Image'])])
X = np.fromstring(X, dtype = np.uint8, sep = ' ')
X = X.reshape((2140, 9216)).astype(dtype = np.float32)
X, y = trans.fit_transform(X), (y - 48) / 48
X_train, y_train = shuffle(X, y, random_state = 42)
X_train = X_train.reshape(-1, 1, 96, 96)

## model

net2 = NeuralNet(

	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('hidden4', layers.DenseLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer)
		], 
	input_shape = (None, 1, 96, 96),
	conv1_num_filters = 32, conv1_filter_size = (3, 3), pool1_ds = (2, 2),
	conv2_num_filters = 64, conv2_filter_size = (2, 2), pool2_ds = (2, 2), 
	conv3_num_filters = 128, conv3_filter_size = (2, 2), pool3_ds = (2, 2), 
	hidden4_num_units = 500,
	hidden5_num_units = 500,
	output_num_units = 30, 
	output_nonlinearity = None,

	update_learning_rate = 0.01, 
	update_momentum = 0.9, 

	regression = True,
	max_epochs = 1000, 
	verbose = 1
	)

net2.fit(X_train, y_train)

