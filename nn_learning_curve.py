# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:54:00 2015

@author: weiwei
"""
# <codecell>
import pandas as pd
import numpy as np
import pickle
from lasagne import layers
from lasagne.nonlinearities import softmax, tanh
from lasagne.updates import nesterov_momentum, sgd
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.decomposition import PCA


# learning curve

def generate_learning_curve(validation_rmse,training_rmse,size_range,model_name):
    
    train, = plt.plot(size_range,training_rmse,label="training",color="blue", marker ="o")
    
    valid, = plt.plot(size_range,validation_rmse,label="validation",color="green", marker ="o")
    
    plt.title(model_name)
    
    plt.xlabel("Training examples")
    
    plt.ylabel("RMSE")
    
    plt.legend(handles = [train,valid],loc="lower left")
    
    plt.savefig(model_name+".png")


trans = MinMaxScaler()

data = pd.read_csv('training.csv', header = 0).dropna()

y = np.array(data[[col for col in data if col != 'Image']]).astype(dtype = np.float32)
X = ' '.join([x for x in np.array(data['Image'])])
X = np.fromstring(X, dtype = np.uint8, sep = ' ')
X = X.reshape((2140, 9216)).astype(dtype = np.float32)
X, y = trans.fit_transform(X), (y - 48) / 48

X_shuffled , y_shuffled = shuffle(X, y, random_state=42)
X_test, y_test = X_shuffled[-140:,:], y_shuffled[-140:,:]
X_train, y_train = X_shuffled[:-140,:], y_shuffled[:-140,:]

"""
trans = MinMaxScaler()

data_train = pd.read_csv('training_nona.csv', header = 0)
#data_train = pd.DataFrame(data_train, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y'])
#data_validation = pd.DataFrame(data_validation, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y'])

X_train, y_train = pickle.load(open('newFeature48by48_train_nona', 'r')).astype(dtype = np.float32), np.array(data_train[[cols for cols in data_train if cols != 'Image']]).astype(dtype = np.float32)
#X_validation, y_validation = pickle.load(open('newFeature48by48_test', 'r')).astype(dtype = np.float32), np.array(data_validation).astype(dtype = np.float32)

X_train, y_train = trans.fit_transform(X_train), (y_train - 48) / 48

# shuffle the data set 
# leave the 140 picture for testing 
X_shuffled, y_shuffled = shuffle(X_train, y_train, random_state=42)
X_test, y_test = X_shuffled[-140:,:], y_shuffled[-140:,:]
X_train, y_train = X_shuffled[:-140,:], y_train[:-140,:]


clf = NeuralNet(

	layers = [
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		#('hidden2', layers.DenseLayer),
		#('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer)
		],

	input_shape = (None, 2304),
	hidden1_num_units = 3,
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
	max_epochs = 400,
	verbose = 1,
	eval_size = 0.2
	)
"""

rmse_list_validation_avg = []

rmse_list_training_avg = []

numExample = np.linspace(.1, 1.0, 10)

for j in numExample:
    
    rmse_list_validation = []
    
    rmse_list_training = []
    
    chosen_idx = np.random.choice(range(len(X_train)),int(j*len(X_train)),replace=False)
    
    x_t, y_t = X_train[chosen_idx], y_train[chosen_idx]  
    
    kf = KFold(len(x_t),n_folds=10)
   
    np.random.seed(10)		
 
    for train_idx, test_idx in kf:
        
        #print len(train_idx),len(test_idx)
        
        train_x, train_y = x_t[train_idx], y_t[train_idx]
        test_x, test_y = x_t[test_idx], y_t[test_idx]
   
	pca = PCA(100)

	pca.fit(train_x)

	train_x = pca.transform(train_x)

	test_x = pca.transform(test_x)	

	clf = NeuralNet(

	layers = [
		('input', layers.InputLayer),
		('hidden1', layers.DenseLayer),
		#('hidden2', layers.DenseLayer),
		#('hidden3', layers.DenseLayer),
		('output', layers.DenseLayer)
		],

	input_shape = (None, 100),
	hidden1_num_units = 20,
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
	max_epochs = 400,
	verbose = 1,
	eval_size = 0.2
	)
     
        clf.fit(train_x,train_y)
        
        pred = clf.predict(test_x)
        
	pred_train = clf.predict(train_x)	

        rmse_list_validation.append(sqrt(mean_squared_error(test_y,pred))*48)
        
        rmse_list_training.append(sqrt(mean_squared_error(train_y,pred_train))*48)
        
    rmse_list_validation_avg.append(np.mean(rmse_list_validation))
    
    rmse_list_training_avg.append(np.mean(rmse_list_training))


#generate_learning_curve(rmse_list_validation_avg, rmse_list_training_avg, numExample, "Learning Curve (Single Layer Neural Network)")

with open("validation_learning","wb") as out_file_valid:
	pickle.dump(rmse_list_validation_avg, out_file_valid)


with open("training_learning","wb") as out_file_train:
        pickle.dump(rmse_list_training_avg, out_file_train)







