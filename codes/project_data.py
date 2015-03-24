import pandas as pd
import numpy as np
import cv2
from scipy import linalg as LA
#from sklearn.decomposition import PCA
import pickle

def show_image(data, num):           # examine the key point from the orginal 96 X 96 image 
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

def show_image2(data, matrix, num):    # examine the key point form the PCA-derived images reduced to 48 X 48 images
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



##  code below is used to calculate covariance matrix, eigenvalues and eigenvectors 
data = pd.read_csv('training_nona.csv', header = 0)



train_matrix = ' '.join([x for x in np.array(data['Image'])])
train_matrix = np.fromstring(train_matrix, dtype = np.uint8, sep = ' ')
train_matrix = train_matrix.reshape((2140, 9216))
train, avg1 = offset(train_matrix)
result1 = np.cov(train, rowvar = 0)
pickle.dump(result1, open('train_covariance_nona', 'wb'))


covariance = pickle.load(open('train_covariance_nona', 'r'))
eigenValue, eigenVector = LA.eigh(np.mat(covariance), eigvals = (6912,9215)) # eigvals parameter will only return the several maximum eigenvalues with
pickle.dump(eigenValue, open('eigenValue48by48_train_nona', 'wb'))                      # corrsponding eigenvectors
pickle.dump(eigenVector, open('eigenVector48by48_train_nona', 'wb'))



# code below is used to calculate the reconstructed image from PCA-derived features 
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
