from scipy import optimize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics
import csv
import numpy as np
import time

''' Test the Least Squares model '''

"""Trust-Region-Reflective Least Squares Algorithm applied to the 
MNIST data set for classification, using stratified K-fold
Cross-Validation. The reason we use the Trust-Region-Reflective Least
Squares Algorithm instead of OLS is that the matrix of explanatory
variables (X) contains mostly zeros. Therefore, it is a non-invertible
(singular) matrix, and we need to use some other optimization algorithm."""

def predict(X_test, beta):
    """Predicts the labels for a given test set and parameter vector beta.

    :param X_test: The test set (containing pixel values) for which to
                   predict labels.
    :param beta: Numpy matrix beta obtained by applying the least squares
                 algorithm to the training set. Each row represents the beta
                 coefficients of a specific digit
    :return: A numby array of label predictions
    """
    
    predictions = [] # array in which we store the predictions

    # For each record, find the highest regression score and the corresponding prediction
    for r in X_test:
        prediction = 0
        maxScore = np.dot(r, beta[0, :])
        for d in range(1, 10):
            score = np.dot(r, beta[d, :])
            if score > maxScore:
                prediction = d
                maxScore = score
        predictions.append(prediction)
        
    return np.array(predictions)



##########---------------MAIN---------------##########

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

# insert bias
X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis = 1)
X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis = 1)
    
print("data loaded")    
    
''' fitting the model '''
start = time.time() # time the fitting for diagnostic purposes
beta_estimates = [] # list in which we store the parameter estimates
    
for d in range(10):
    y_current_digit = np.zeros(y_train.shape[0])
    for i in range(y_train.shape[0]):
        y_current_digit[i] = 1 if y_train[i] == d else 0
        
    solution = optimize.lsq_linear(X_train, y_current_digit) # fit least squares model
    beta_estimates.append(solution.x) # save parameter estimates
beta_estimates = np.array(beta_estimates)
end = time.time()
    
# calculate total fitting time
fit_time = end - start
print("Fitting time: ", fit_time) # print total fitting time

''' evaluate predictive power '''
start = time.time() # time the predictions for diagnostic purposes
y_pred = predict(X_test, beta_estimates) # predict values for testing set
end = time.time()
    
# calculate total prediction time
pred_time = end - start
print("Prediction time: ", pred_time) # print total fitting time
    
# calculate error rate
error_rate = 1 - metrics.accuracy_score(y_test, y_pred)
print("Error rate: ", error_rate)

