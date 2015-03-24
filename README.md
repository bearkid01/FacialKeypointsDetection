# FacialKeypointsDetection
Group Project of Advanced Machine Learning (MSAN 630)

The purpose of this project is to apply advanced machine learning techniques to automatically detect facial keypoints on a given facial image. We are focusing on 2D images for this project. Our machine learning approach starts from extracting important features from a high dimensional data space using principal component analysis. These features are then fed into a model to predict keypoints on previously unseen facial images. Two algorithms are investigated to build our models: a  neural network model with single hidden layer, our baseline solution and a deep learning technique, namely convolutional neural network (CNN), our advanced solution. We evaluate our approach by calculating the root mean square error (RMSE) between the predicted and actual positions of the keypoints. Our aim is to minimize RMSE.

Without using any technique from computer vision, we proved that a single hidden layer neural network could easily approximate the underlying function. Also, our decision to choose neural networks at the first place is sound. Both models fit the characteristics of the datasets, taking a high-dimensional input and outputting a numeric vector. Compared with the single layer neural network, our advanced solution, the convolutional neural network reached an even lower RMSE on the test dataset without being cross-validated. Again, the results from Kaggle demonstrate the strong capability of the convolutional neural network in image recognition. 

Here are nine images selected from the test dataset marked with facial keypoints outputted by our CNN model. 
![Images With Facial Keypoints Marked](https://github.com/bearkid01/FacialKeypointsDetection/blob/master/graphs/3by3.png)
