
import csv

''' This script can be used to combine the images data and the labels into one full data set '''

# open image data
with open('mnist_image_test.csv', 'r') as imgs:
    images = [[int(x) for x in rec] for rec in csv.reader(imgs, delimiter=',')]

# open label data 
with open('mnist_label_test.csv', 'r') as lbls:
    labels = [[int(x) for x in rec] for rec in csv.reader(lbls, delimiter=',')]


# combine the images and labels into one dataset
with open('mnist_full_test.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for i in range(len(labels)):
        image=images[i]
        newRow = [labels[i][0]]
        for i in image:
            newRow.append(int(float(i)))
        writer.writerow(newRow)