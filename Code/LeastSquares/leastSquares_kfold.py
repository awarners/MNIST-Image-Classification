from scipy import optimize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics
import csv
import numpy as np
import time

''' K-Fold Cross-Validation using a Least Squares model. '''

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
# Import the data file:
with open('mnist_full_train.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data = list(reader)
data = np.array(data)

print("data loaded")

# Split input and target values
X_data, y_data = data[:, 1:], data[:, 0]

# Make arrays for results
error_rates = []
fitting_times = []
prediction_times = []

# set up cross-validation procedure
cross_validator = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

iteration = 1
for train_index, test_index in cross_validator.split(X_data, y_data):
    
    print("\n iteration: ", iteration)
    
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    # insert bias values
    X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis = 1)
    X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis = 1)
    
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
    fitting_times.append(fit_time)
    print("Fitting time: ", fit_time) # print total fitting time
    
    ''' evaluate predictive power '''
    start = time.time() # time the predictions for diagnostic purposes
    y_pred = predict(X_test, beta_estimates) # predict values for testing set
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

