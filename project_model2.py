import pandas as pd
import numpy as np
import pickle
from lasagne import layers
from lasagne.nonlinearities import softmax, tanh
from lasagne.updates import nesterov_momentum, sgd
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 


# data input

trans = MinMaxScaler()



data_train = pd.read_csv('training_nona.csv', header = 0)
#data_train = pd.DataFrame(data_train, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y'])
#data_validation = pd.DataFrame(data_validation, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y'])



X_train, y_train = pickle.load(open('newFeature48by48_train_nona', 'r')).astype(dtype = np.float32), np.array(data_train[[cols for cols in data_train if cols != 'Image']]).astype(dtype = np.float32)
#X_validation, y_validation = pickle.load(open('newFeature48by48_test', 'r')).astype(dtype = np.float32), np.array(data_validation).astype(dtype = np.float32)

X_train, y_train = trans.fit_transform(X_train), (y_train - 48) / 48


model = NeuralNet(

	layers = [
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		#('hidden2', layers.DenseLayer),
		#('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer)
		],

	input_shape = (None, 2304),
	hidden1_num_units = 30,
	#hidden1_nonlinearity = softmax,
	#hidden2_num_units = 20,
	#hidden3_num_units = 5,
	#hidden2_nonlinearity = tanh,
	output_nonlinearity = None,
	output_num_units = 30,

	update = nesterov_momentum,
	update_learning_rate = 0.025,
	update_momentum = 0.9,

	regression = True,
	max_epochs = 500,
	verbose = 1,
	eval_size = 0.2
	)

model.fit(X_train, y_train)
train_loss = np.array([np.sqrt(i["train_loss"]) * 48 for i in model.train_history_])
valid_loss = np.array([np.sqrt(i["valid_loss"]) * 48 for i in model.train_history_])
plt.plot(train_loss, linewidth = 3, label = 'train')
plt.plot(valid_loss, linewidth = 3, label = 'valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.ylim(2.5, 5)
plt.show()

