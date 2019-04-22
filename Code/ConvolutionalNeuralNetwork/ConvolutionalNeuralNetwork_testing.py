from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from sklearn import metrics
import csv
import numpy as np
import time

''' Test the Convolutional Neural Network model '''

print("running...")
# Import the train data file:
with open('deskewed_full_train.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data1 = list(reader)
data = np.array(data1)

with open('mnist_full_train.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data2 = list(reader)
data = np.concatenate( (data1, np.array(data2)), axis = 0 )

# Split input and target values
X_train, y_train = data[:, 1:], data[:, 0]

# Import the test data file:
with open('deskewed_full_test.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data = list(reader)
data = np.array(data)

# Split input and target values
X_test, y_test = data[:, 1:], data[:, 0]
    
print("data loaded")    

''' define the model '''
clf = ConvolutionalNeuralNetwork(hidden_layers=[500, 400, 300], filter_shape = [5, 5], num_filter_layers = 15, pooling_parameter = 3, epochs = 1, seed=0, batch_size = 64, learning_rate = 0.3, 
                    min_learning_rate = 0.05, learning_rate_decay = 0.95, momentum_factor = 0.0, leaky_factor = 0.0, 
                    deactivation_prob = [0.0, 0.5, 0.4, 0.3])

''' fitting the model '''
start = time.time() # time the fitting for diagnostic purposes
clf.fit(X_train, y_train) # build/fit tree
end = time.time()
    
# calculate total fitting time
fit_time = end - start
print("Fitting time: ", fit_time) # print total fitting time

''' evaluate predictive power '''
start = time.time() # time the predictions for diagnostic purposes
y_pred = clf.predict(X_test) # predict values for testing set
end = time.time()
    
# calculate total prediction time
pred_time = end - start
print("Prediction time: ", pred_time) # print total fitting time
    
# calculate error rate
error_rate = 1 - metrics.accuracy_score(y_test, y_pred)
print("Error rate: ", error_rate)
