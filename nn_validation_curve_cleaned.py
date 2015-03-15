# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:35:17 2015

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
from sklearn.decomposition import PCA

trans = MinMaxScaler()

data = pd.read_csv('training.csv', header = 0).dropna()

y = np.array(data[[col for col in data if col != 'Image']]).astype(dtype = np.float32)
X = ' '.join([x for x in np.array(data['Image'])])
X = np.fromstring(X, dtype = np.uint8, sep = ' ')
X = X.reshape((2140, 9216)).astype(dtype = np.float32)
X, y = trans.fit_transform(X), (y - 48) / 48
#X_train, y_train = shuffle(X, y, random_state = 42)

# validation curve 

# shuffle the data set 
# leave the 140 picture for testing 
X_shuffled , y_shuffled = shuffle(X, y, random_state=42)
X_test, y_test = X_shuffled[-140:,:], y_shuffled[-140:,:]
X_train, y_train = X_shuffled[:-140,:], y_shuffled[:-140,:]


rmse_list_validation_avg = []

rmse_list_training_avg = []

numHiddenUnits_epochs = [1,5,10,20,30,50,100]

for n in numHiddenUnits_epochs:
    
    rmse_list_validation = []
    
    rmse_list_training = []
   
    np.random.seed(40)
 
    kf = KFold(len(X_train),n_folds=10)
    
    for train_idx, test_idx in kf:
        
        pca = PCA(100)

        train_x, train_y = X_train[train_idx], y_train[train_idx]
        
        
        pca.fit(train_x)
        
        train_x = pca.transform(train_x)

        
        
        test_x, test_y = X_train[test_idx], y_train[test_idx]
        
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
    	       hidden1_num_units = n,
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
        
        rmse_list_training.append(np.sqrt(mean_squared_error(train_y,pred_train))*48)
        
        
        rmse_list_validation_avg.append(np.mean(rmse_list_validation))
    
        rmse_list_training_avg.append(np.mean(rmse_list_training))
    


with open("validation","wb") as out_file_valid:
	pickle.dump(rmse_list_validation_avg,out_file_valid)

with open("training","wb") as out_file_train:
	pickle.dump(rmse_list_training_avg,out_file_train)
#generate_validation_curve(validation,train,Num,"Validation Curve (Single Layer Neural Network)")
    
  



