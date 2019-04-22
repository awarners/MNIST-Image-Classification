from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics
import csv
import numpy as np
import time

''' Test the Random Forest model '''

print("running...")
# Import the train data file:
with open('mnist_full_train.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data1 = list(reader)
data = np.array(data1)

# Split input and target values
X_train, y_train = data[:, 1:], data[:, 0]

# Import the test data file:
with open('mnist_full_test.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data = list(reader)
data = np.array(data)

# Split input and target values
X_test, y_test = data[:, 1:], data[:, 0]
    
print("data loaded")    
    
''' define the model '''
clf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', max_depth = 50, min_samples_split = 5, n_jobs = -1)

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
