# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:55:49 2015

@author: weiwei
"""

# <codecell>
import matplotlib 
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage import filter


matplotlib.rcParams['font.size'] = 9


image = camera()
thresh = filter.threshold_otsu(image)
binary = image > thresh


# <codecell>
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(image)
ax2.set_title('Histogram')
ax2.axvline(thresh, color='r')

ax3.imshow(binary, cmap=plt.cm.gray)
ax3.set_title('Thresholded')
ax3.axis('off')

plt.show()
    
    
# <codecell>
    
# make the TRUE to 0 and keep False 
    
image_test =image[100]
binary_test = binary[100]


# <codecell>
updated_image = [0 if binary_test[i] == True else image_test[i] for i in range(len(image_test))]
# <codecell>

updated_image = [[0 if binary[j][i] == True else image[j][i] for i in range(len(image[j]))] for j in range(len(image))]


# <codecell>

updated_image_nparray = np.array(updated_image)

# <codecell>

plt.imshow(updated_image_nparray,cmap=plt.cm.gray)

# <codecell>

# try one picture for now 
import pandas as pd
training = pd.read_csv("training.csv")
pixels = training.ix[:,30]

# <codecell>
pixels_together = pixels.tolist()

# <codecell>
image_1 = pixels_together[0]
# <codecell>

image_1_list = image_1.split(" ")

# <codecell>
image_1_matrix = np.array(image_1_list).reshape(96,96)

# <codecell>
image_1_matrix = image_1_matrix.astype(int)

# <codecell>
image_1_matrix

# <codecell>
thresh_1 = filter.threshold_otsu(image_1_matrix)

# <codecell>
binary = image_1_matrix > thresh_1

# <codecell>

updated_image_1 = [[0 if binary[j][i] == True else image_1_matrix[j][i] for i in range(len(image_1_matrix[j]))] for j in range(len(image_1_matrix))]

# <codecell>
updated_image_1_updated = np.array(updated_image_1)

# <codecell>

plt.figure(1)
plt.subplot(211)
plt.imshow(updated_image_1_updated)
plt.subplot(212)
plt.imshow(image_1_matrix)

# <codecell>

# <codecell>
image_2 = pixels_together[1]
# <codecell>

image_2_list = image_2.split(" ")

# <codecell>
image_2_matrix = np.array(image_2_list).reshape(96,96)



# <codecell>
image_2_matrix = image_2_matrix.astype(int)
plt.imshow(image_2_matrix)

# <codecell>
image_2_matrix

# <codecell>
thresh_2 = filter.threshold_otsu(image_2_matrix)

# <codecell>
binary = image_2_matrix > thresh_2
plt.imshow(binary)

# <codecell>

updated_image_2 = [[0 if binary[j][i] == True else image_2_matrix[j][i] for i in range(len(image_2_matrix[j]))] for j in range(len(image_2_matrix))]

# <codecell>
updated_image_2_updated = np.array(updated_image_2)

# <codecell>

plt.figure(1)
plt.subplot(211)
plt.imshow(updated_image_2_updated,cmap=plt.cm.gray)
plt.subplot(212)
plt.imshow(image_2_matrix,cmap=plt.cm.gray)


# <codecell>
image_9 = pixels_together[8]
# <codecell>

image_9_list = image_9.split(" ")

# <codecell>
image_9_matrix = np.array(image_9_list).reshape(96,96)

# <codecell>
image_9_matrix = image_9_matrix.astype(int)
plt.imshow(image_9_matrix)

# <codecell>
image_9_matrix

# <codecell>
thresh_9 = filter.threshold_otsu(image_9_matrix)

# <codecell>
binary = image_9_matrix > thresh_9
plt.imshow(binary)

# <codecell>

updated_image_9 = [[0 if binary[j][i] == True else image_9_matrix[j][i] for i in range(len(image_9_matrix[j]))] for j in range(len(image_9_matrix))]

# <codecell>
updated_image_9_updated = np.array(updated_image_9)

# <codecell>

plt.figure(1)
plt.subplot(211)
plt.imshow(updated_image_9_updated,cmap=plt.cm.gray)
plt.subplot(212)
plt.imshow(image_9_matrix,cmap=plt.cm.gray)


# <codecell>



# <codecell>