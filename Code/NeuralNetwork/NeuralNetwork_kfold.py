from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics
import csv
import numpy as np
import time

''' K-Fold Cross-Validation using the Neural Network model. '''

print("running...")
# Import the data file:
with open('deskewed_full_train.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data1 = list(reader)
data = np.array(data1) 
    
# Split input and target values
X_data, y_data = data[:, 1:], data[:, 0]

print("data loaded")   

# Make arrays for results
error_rates = []
fitting_times = []
prediction_times = []

# set up cross-validation procedure
cross_validator = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

''' define the model '''
clf = NeuralNetwork(hidden_layers=[500, 400, 300], epochs = 30, seed=0, batch_size = 64, learning_rate = 0.3, 
                    min_learning_rate = 0.05, learning_rate_decay = 0.95, leaky_factor = 0.0, 
                    deactivation_prob = [0.0, 0.5, 0.4, 0.3])

print("start cross-validation")

iteration = 1
for train_index, test_index in cross_validator.split(X_data, y_data):
    
    print("\n iteration: ", iteration)
    
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    ''' fitting the model '''
    start = time.time() # time the fitting for diagnostic purposes
    clf.fit(X_train, y_train) # build/fit tree
    end = time.time()
    
    # calculate total fitting time
    fit_time = end - start
    fitting_times.append(fit_time)
    print("Fitting time: ", fit_time) # print total fitting time

    ''' evaluate predictive power '''
    start = time.time() # time the predictions for diagnostic purposes
    y_pred = clf.predict(X_test) # predict values for testing set
    end = time.time()
    
    # calculate total prediction time
    pred_time = end - start
    prediction_times.append(pred_time)
    print("Prediction time: ", pred_time) # print total fitting time
    
    # calculate error rate
    error_rate = 1 - metrics.accuracy_score(y_test, y_pred)
    error_rates.append(error_rate)
    print("Error rate: ", error_rate)
    
    iteration += 1


# print results
print("\n Average fitting time: ", np.mean(fitting_times))
print("Average prediction time: ", np.mean(prediction_times))
print("Average error rate: ", np.mean(error_rates))

