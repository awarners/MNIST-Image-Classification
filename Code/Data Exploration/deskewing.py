from scipy.ndimage import interpolation
import numpy as np
import csv
import time

''' This scripts can be used to deskew the entire data set of MNIST images.
    Source of code is https://fsix.github.io/mnist/Deskewing.html, however it has been slightly modified
    Authors: Alvin Wan, Dibya Ghosh and Siqi Liu
    Year: 2016
    Accessed: 12-03-2019'''

# function that deskews an image
def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    img = interpolation.affine_transform(image,affine,offset=offset)
    return 255*(img - img.min()) / (img.max() - img.min())  # Normalize values and return

# function that calculates the moments of an image
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    mean0 = np.sum(c0*image)/totalImage #mu_x
    mean1 = np.sum(c1*image)/totalImage #mu_y
    cov00 = np.sum((c0-mean0)**2*image)/totalImage #var(x)
    cov11 = np.sum((c1-mean1)**2*image)/totalImage #var(y)
    cov01 = np.sum((c0-mean0)*(c1-mean1)*image)/totalImage #covariance(x,y)
    mean_vector = np.array([mean0,mean1])
    covariance_matrix = np.array([[cov00,cov01],[cov01,cov11]])
    return mean_vector, covariance_matrix

# Choose which datasets to open:
with open('mnist_image_test.csv', 'r') as imgs:
    images = [[int(x) for x in rec] for rec in csv.reader(imgs, delimiter=',')]
    
with open('mnist_label_test.csv', 'r') as lbls:
    labels = [[int(x) for x in rec] for rec in csv.reader(lbls, delimiter=',')]

average= 0
# Deskew the data and create a new .csv file to store it in
with open('deskewed_image_test.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for i in range(len(labels)):
        start = time.time()
        deskewed = deskew(np.reshape(images[i], (28,28)))
        average += time.time()-start
        image = np.reshape(deskewed, (784, 1))
        
        image = [int(float(i)) for i in image]
        writer.writerow(image)

print(average / 10000)    
    
    
