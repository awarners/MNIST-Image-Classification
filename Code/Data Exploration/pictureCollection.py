

import matplotlib.pyplot as plt
import numpy
from PIL import Image
import csv

''' This script can be used to create the collections of pictures as seen in the paper '''

# Choose which datasets to inspect and open it
with open('deskewed_image_train.csv', 'r') as imgs:
    images = [[int(x) for x in rec] for rec in csv.reader(imgs, delimiter=',')]
    
with open('mnist_label_train.csv', 'r') as lbls:
    labels = [[int(x) for x in rec] for rec in csv.reader(lbls, delimiter=',')]

digit = 0; # keeps track of which digit it's at
count = 0; # keeps track of how many pictures have already been stored for the current digit
data = numpy.zeros((280, 280, 3), dtype = numpy.uint8) # stores all desired pixel values

while digit < 10:
    i = 0 # index to keep track of where it is in the pictures database
    while count < 10:
        if (labels[i][0] == digit):
            image = images[i] # reference to new picture
            
            # Copy the picture
            for x in range(28):
                for y in range(28):
                    data[digit*28+y, count*28+x] = 255 - image[28*y + x]
            count=count+1
            
        i=i+1 
    digit=digit+1;
    count=0;
    
    

newImage = Image.fromarray(data) # Create image
fn = "collection_deskewed.png"
newImage.save(fn) # Save the new picture