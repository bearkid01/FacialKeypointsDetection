# <codecell>
import pandas as pd
import numpy as np
import pickle
from lasagne import layers
from lasagne.nonlinearities import softmax, tanh
from lasagne.updates import nesterov_momentum, sgd
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt 
from sklearn.utils import shuffle

# data input
"""
trans = MinMaxScaler()

data_train = pd.read_csv('training_nona.csv', header = 0)
#data_train = pd.DataFrame(data_train, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y'])
#data_validation = pd.DataFrame(data_validation, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y'])

X_train, y_train = pickle.load(open('newFeature48by48_train_nona', 'r')).astype(dtype = np.float32), np.array(data_train[[cols for cols in data_train if cols != 'Image']]).astype(dtype = np.float32)
#X_validation, y_validation = pickle.load(open('newFeature48by48_test', 'r')).astype(dtype = np.float32), np.array(data_validation).astype(dtype = np.float32)

X_train, y_train = trans.fit_transform(X_train), (y_train - 48) / 48

# <codecell>

training = pd.read_csv("training.csv").dropna()
trainingImages = training["Image"]

pixels_together = trainingImages.tolist()

# use map function to break up the string 
pixels_together_updated = map(lambda x: x.split(" "), pixels_together)

pixels_tegether_updated_2 = [map(lambda x:int(x),i) for i in pixels_together_updated]

colnames = list(training.columns.values)
colnames.remove("Image")

coords = training[colnames]

X_train_ori = np.array(pixels_tegether_updated_2)

y_train_ori = np.array(coords)

"""

trans = MinMaxScaler()

data = pd.read_csv('training.csv', header = 0).dropna()

y = np.array(data[[col for col in data if col != 'Image']]).astype(dtype = np.float32)
X = ' '.join([x for x in np.array(data['Image'])])
X = np.fromstring(X, dtype = np.uint8, sep = ' ')
X = X.reshape((2140, 9216)).astype(dtype = np.float32)
X, y = trans.fit_transform(X), (y - 48) / 48
X_train, y_train = shuffle(X, y, random_state = 42)
X_train = X_train.reshape(-1, 1, 96, 96)

test = pd.read_csv('test.csv',header=0)
test_X = ' '.join([x for x in np.array(test['Image'])])
test_X = np.fromstring(test_X, dtype = np.uint8, sep = ' ')
test_X = test_X.reshape((1783, 9216)).astype(dtype = np.float32)
#X, y = trans.fit_transform(X), (y - 48) / 48
#X_train, y_train = shuffle(X, y, random_state = 42)
test_X = trans.fit_transform(test_X)
test_X = test_X.reshape(-1,1,96,96)

# <codecell>
"""
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
 

# <codecell>
model.fit(X_train, y_train) 



# <codecell>
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
# <codecell>
# convolutional neural nets 

convX = X_train_ori.reshape(-1, 1, 96, 96)
convy = y_train_ori

# <codecell>

"""

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

	update_learning_rate = 0.02, 
	update_momentum = 0.9, 

	regression = True,
	max_epochs = 500, 
	verbose = 1
	)


net2.fit(X_train,y_train)
pred = net2.predict(test_X)

with open("pred","wb") as out_file:
	pickle.dump(pred,out_file)


with open("history","wb") as hist_out_file:
	pickle.dump(net2.train_history_,hist_out_file)



"""
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
"""

