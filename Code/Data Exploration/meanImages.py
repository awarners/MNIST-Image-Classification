
from mnist import MNIST
from PIL import Image
import random
import numpy
import math
import csv

''' This script can be used to create the mean images as seen in the paper '''

# Choose which datasets to inspect and open it
with open('deskewed_image_train.csv', 'r') as imgs:
    images = [[int(x) for x in rec] for rec in csv.reader(imgs, delimiter=',')]
    
with open('mnist_label_train.csv', 'r') as lbls:
    labels = [[int(x) for x in rec] for rec in csv.reader(lbls, delimiter=',')]

# Count the amount of examples available for each digit
digitCounts = numpy.zeros(10)
for i in range(len(labels)):
    digitCounts[labels[i]] = digitCounts[labels[i]] + 1

# Process the mean picture for each digit
for d in range(10):
    data = numpy.zeros((28, 28, 3)) # List in which we can store the average pixel values
    
    # Process each training example that has the label of the current digit
    for i in range(len(labels)):
        if labels[i][0] == d:
            image = images[i]
            for x in range(28):
                for y in range(28):
                    data[y, x] = data[y, x] + (255 - image[28*y + x])
        
    # Calculate averages
    for x in range(28):
        for y in range(28):
            data[y, x] = round(data[y, x, 0]/digitCounts[d])


    data = data.astype(numpy.uint8) # Convert pixel values to integers
    newImage = Image.fromarray(data) # Create image
    fn = "mean" + str(d) + ".png" # New file name
    newImage.save(fn) # Save the new picture
    
    print("Count for digit " + str(d) + ": " + str(digitCounts[d]))