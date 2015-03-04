import pandas as pd
import numpy as np
import cv2
from scipy import linalg as LA
#from sklearn.decomposition import PCA
import pickle

def show_image(data, num):
	lefteye = int(data.loc[num, 'left_eye_center_x']),int(data.loc[num, 'left_eye_center_y']) 
	righteye = int(data.loc[num, 'right_eye_center_x']),int(data.loc[num, 'right_eye_center_y'])
	image = np.fromstring(data['Image'][num], dtype = np.uint8, sep = ' ').reshape((96,96))
	#image =  np.array([int(x) for x in data['Image'][num].split()], dtype = np.uint8).reshape((96,96))
	image = cv2.circle(image, lefteye, 1, (255,255,255))
	image = cv2.circle(image, righteye, 1, (255,255,0))
	image = cv2.resize(image, (0,0), fx = 4, fy = 4)
	cv2.imwrite('96X96.png', image)
	cv2.imshow('Image', image)
	cv2.waitKey()

def show_image2(data, matrix, num):
	lefteye = int(data.loc[num, 'left_eye_center_x']),int(data.loc[num, 'left_eye_center_y']) 
	righteye = int(data.loc[num, 'right_eye_center_x']),int(data.loc[num, 'right_eye_center_y'])
	image = np.array(matrix[num,], dtype = np.uint8).reshape((96, 96))
	image = cv2.circle(image, lefteye, 1, (255, 255, 255))
	image = cv2.circle(image, righteye, 1, (255, 255, 0))
	image = cv2.resize(image, (0, 0), fx = 4, fy = 4)
	cv2.imwrite('48X48.png', image)
	cv2.imshow('Image', image)
	cv2.waitKey()
###  ----- above function just for visulization 

###  below three functions are for PCA calculation 

def offset(data):    #  remove mean values of original data
	avg = np.mean(data, axis = 0)
	newData = data - avg
	return newData, avg

def percentageCal(eigenValue, percentage):      # 
	eigenValue = np.sort(eigenValue)[-1::-1]
	eigenValueSum = np.sum(eigenValue)
	temp, num = 0, 0
	for x in eigenValue:
		temp += x
		num += 1
		if temp >= eigenValueSum * percentage:
			break
	return num

def pcaCal(data, percentage = 0.8):
	newData, avg = offset(data)
	covCal = np.cov(newData, rowvar = 0)      # this step take about 10 minutes to run, so, I save covariance matrix as a file
	#eigenValue, eigenVector = np.linalg.eig(np.mat(covCal))
	eigenValue, eigenVector = LA.eig(np.mat(covCal))  # I did not test, but I think this will also take very long time to run
	n = percentageCal(eigenValue, percentage)         # insteand I used scipy.linalg.eigh to calculate. since covariance matrix is symmetric
	num_eigenValueIndice = np.argsort(eigenValue)[-1:-(n+1):-1]  # can use eigvals parameter
	new_eigenVector = eigenVector[:, num_eigenValueIndice]
	newFeature = newData.dot(new_eigenVector)
	reconFeature = (newFeature.dot(new_eigenVector.T)) + avg
	return newFeature, reconFeature



'''
np.random.seed(seed = 42)
data = pd.read_csv('training.csv', header = 0)
train_idx = np.random.choice(data.shape[0], size = 5640, replace = False)
test_idx = [x for x in range(data.shape[0]) if x not in train_idx]
train_data, test_data = data.loc[train_idx], data.loc[test_idx]
#train_data.to_csv('train.csv', index = False)
#test_data.to_csv('test.csv', index = False)
'''
#data = pd.read_csv('training_nona.csv', header = 0)

'''
train_data, test_data = pd.read_csv('train.csv', header = 0), pd.read_csv('test.csv', header = 0) 
train_data = pd.DataFrame(train_data, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y', 'Image']).dropna()
test_data = pd.DataFrame(test_data, columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y', 'Image']).dropna()
train_data.to_csv('train2.csv', index = False)
test_data.to_csv('test2.csv', index = False)
'''
'''
train_matrix, test_matrix = ' '.join([x for x in np.array(train_data['Image'])]), ' '.join([x for x in np.array(test_data['Image'])])
train_matrix, test_matrix = np.fromstring(train_matrix, dtype = np.uint8, sep = ' '), np.fromstring(test_matrix, dtype = np.uint8, sep = ' ')
train_matrix, test_matrix = train_matrix.reshape((5626, 9216)), test_matrix.reshape((1407, 9216))
'''
'''
train_matrix = ' '.join([x for x in np.array(data['Image'])])
train_matrix = np.fromstring(train_matrix, dtype = np.uint8, sep = ' ')
train_matrix = train_matrix.reshape((2140, 9216))
train, avg1 = offset(train_matrix)
result1 = np.cov(train, rowvar = 0)
pickle.dump(result1, open('train_covariance_nona', 'wb'))
'''
'''
test, avg2 = offset(test_matrix)
result2 = np.cov(test, rowvar = 0)
pickle.dump(result2, open('test_covariance', 'wb'))
'''
'''
#matrix = ' '.join([x for x in np.array(data['Image'])])       # combine each one string to one big string for the value of target variable data['Image']
#matrix = np.fromstring(matrix, dtype = np.uint8, sep = ' ')   # read the string and to form a matrix



matrix = matrix.reshape((7049, 9216))                         # reshape the matrix to 7049 rows and 9216 cols
newData, avg = offset(matrix)                                 # preprocessing for covariance calculation --- normalization ( remove mean )



#newFeature, reconFeature = pcaCal(matrix)
newData, avg = offset(matrix)
result = np.cov(newData, rowvar = 0)
pickle.dump(result, open('covariance', 'wb'))
'''
'''
covariance = pickle.load(open('train_covariance_nona', 'r'))
eigenValue, eigenVector = LA.eigh(np.mat(covariance), eigvals = (6912,9215)) # eigvals parameter will only return the several maximum eigenvalues with
pickle.dump(eigenValue, open('eigenValue48by48_train_nona', 'wb'))                      # corrsponding eigenvectors
pickle.dump(eigenVector, open('eigenVector48by48_train_nona', 'wb'))
'''



data = pd.read_csv('training_nona.csv', header = 0)

matrix = ' '.join([x for x in np.array(data['Image'])])
matrix = np.fromstring(matrix, dtype = np.uint8, sep = ' ') 
matrix = matrix.reshape((2140, 9216))
newData, avg = offset(matrix) 
eigenvectors = pickle.load(open('eigenVector48by48_train_nona', 'r'))
newFeature = newData.dot(eigenvectors)
pickle.dump(newFeature, open('newFeature48by48_train_nona', 'wb'))


reconFeature = (newFeature.dot(eigenvectors.T)) + avg   # reconstruct the features in orginal feature space
#print reconFeature.shape
show_image2(data, reconFeature, 888)
