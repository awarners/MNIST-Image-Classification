
import numpy as np
import math
import time
from sklearn import metrics

''' Convolutional Neural Network implementation that uses ReLu functions for the hidden layers and filters, and the softmax function 
    for the output. This implementation uses mini-batch stochastic gradient descent (MBSGD) and allows for momentum, decaying learning rates,
    leaky ReLu and Dropout. This implementation only uses 1 layer of convolutions, activations and pooling.'''

class ConvolutionalNeuralNetwork:

    def __init__(self, hidden_layers, filter_shape = [3, 3], num_filter_layers = 10, pooling_parameter = 2, epochs = 100, seed = 0, batch_size = 100,
                  learning_rate = 0.05, min_learning_rate = 0.01, learning_rate_decay = 1.0, momentum_factor = 0.0, leaky_factor = 0.0,
                   deactivation_prob = -1):
        
        """Constructor for the Neural Network
        :param hidden_layers: list defining the structure of the hidden layers
        :param filter_shape: list defining the size of the filters (kernels)
        :param num_filter_layers: int that determines how many filters the model uses
        :param pooling_parameter: int that determines the size and stride of the pooling window
        :param epochs: int that determines how many epochs the model should train for
        :param seed: int that acts as the seed for the random number generation
        :param batch_size: int that determines the size of the mini-batches for MBSGD
        :param learning_rate: double that determines the step size during MBSGD
        :param learning_rate: double that determines the lowest the learning rate can decay to
        :param learning_rate_decay: double that determines by how much we decrease the learning rate after each epoch
        :param momentum_factor: double that determines the step sizes of the momentum updates
        :param leaky_factor: double that determines the leak of the ReLu function
        :param deactivation_prob: list defining deactivation probabilities for every layer except the output layer. Set to -1 if Dropout is not wanted.
        """
        
        self.hidden_layers = hidden_layers
        self.filter_shape = filter_shape
        self.num_filter_layers = num_filter_layers
        self.pooling_parameter = pooling_parameter
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.momentum_factor = momentum_factor
        self.leaky_factor = leaky_factor
        
        if deactivation_prob == -1 or len(deactivation_prob) != len(hidden_layers) + 1:
            self.deactivation_prob = np.zeros(len(hidden_layers) + 1)
        else:
            self.deactivation_prob = deactivation_prob
        
        self.weights = []
    
    def fit(self, X_data, y_data):
        """Trains the Neural Network
        :param X_data: NumPy matrix of input records. Each row represents one record and the columns represent features.
        :param y_data: NumPy matrix(vector) of labels. Each element is is a separate label.
        """
        
        layers = self.hidden_layers
        
        # normalize training data
        X_data = self.normalize_training_data(X_data)

        num_inputs = int( (math.sqrt(X_data.shape[1]) - self.filter_shape[0] + 1)*(math.sqrt(X_data.shape[1]) - self.filter_shape[1] + 1) / self.pooling_parameter**2 * self.num_filter_layers) # amount of inputs
        num_outputs = np.unique(y_data).shape[0] # amount of outputs
        num_records = y_data.shape[0] # amount of records
        num_layers = len(layers) # amount of hidden layers
        
        # initialize weights and filters:
        self.initialize_weights(num_inputs, num_outputs)
        self.initialize_filters()
        
        ''' improve weights till max amount of epochs is reached '''
        for epoch in range(self.epochs):
            
            start= time.time()
            print("Epoch: ", epoch + 1)
            
            # divide the records into random mini-batches
            mini_batch_indices = np.random.choice(num_records, size = (math.floor(num_records / self.batch_size), self.batch_size), replace = False)
            
            # last update to weights - used for momentum
            prev_d_weights = None
            prev_d_filters = None
            prev_d_biases = None
            
            ''' process each batch '''
            for batch in mini_batch_indices:
                
                # initialize matrices to store the weight changes in
                d_weights = []
                d_filters = np.zeros(self.filters.shape)
                d_biases = np.zeros(self.biases.shape)
                
                # forward pass the current batch
                batch_activations, batch_convolutions = self.forward_pass(X_data[batch].copy())
                batch_labels = y_data[batch]
                
                # calculate error for each output unit
                error = batch_activations[num_layers + 1].copy()
                for i in range(error.shape[0]):
                    error[i, int(batch_labels[i])] -= 1
                
                # variable that holds the value of derror_dout * dout_din
                carry_over = 0

                ''' do backpropagation for entire batch '''
                for i in range(num_layers + 1):
                        
                    layer_i = num_layers - i # actual layer index
                        
                    prev_layer_activations = batch_activations[layer_i] # activations of previous layer
                    current_layer_activations = batch_activations[layer_i + 1] # activations of current layer

                    if layer_i == num_layers: # output layer
                        carry_over = error # softmax + crossentropy loss gives this nice gradient
                            
                    else: # all other layers
                        carry_over = np.matmul(self.weights[layer_i + 1], np.transpose(carry_over)) # error at the neurons of the 'next' layer

                        carry_over = np.transpose(carry_over)
                        
                        carry_over[current_layer_activations <= 0] *= self.leaky_factor # ReLu gradient
                    
                    if layer_i < num_layers: # remove bias errors as there is no connection between the previous layer and the bias in the next
                            carry_over = carry_over[:, 1:]
            
                    # WARNING: transforming the data type to 32 bit decreases the computation times, but it also decreases weight precision!
                    carry_over = carry_over.astype(dtype ='float32')
                    prev_layer_activations = prev_layer_activations.astype(dtype ='float32')
                    
                    # calculate the update for this layer
                    d_weights.append(np.einsum('ij,ik->jk', prev_layer_activations, carry_over)) # calculate updates according to this record
                    
                # backpropagation through convolutional layers

                carry_over = np.matmul(self.weights[0], np.transpose(carry_over)) # error at the neurons of the 'next' layer

                carry_over = np.transpose(carry_over)
                    
                carry_over = carry_over[:, 1:] # remove bias errors as there is no connection between the previous layer and the bias in the next
                
                # reshape error matrix into actual image matrix instead of vector of features
                carry_over = carry_over.reshape(carry_over.shape[0], self.num_filter_layers, int( (math.sqrt(X_data.shape[1]) - self.filter_shape[0] + 1) / self.pooling_parameter), int( (math.sqrt(X_data.shape[1]) - self.filter_shape[1] + 1) / self.pooling_parameter))
                
                # add rows and columns to error matrix to get the right size for element-wise matrix multiplication
                carry_over = np.repeat(carry_over, self.pooling_parameter, 2)
                carry_over = np.repeat(carry_over, self.pooling_parameter, 3)
                
                output_booleans = batch_convolutions[2]
                
                d_matrix = output_booleans * carry_over
                
                # actual input image
                inputs = batch_convolutions[0]
                
                # obtain all the submatrices from the images that need to be convolved
                filter = self.filter_shape
                r,x,y = inputs.shape
                sr, sx, sy = inputs.strides    
                nrows = x-filter[0]+1
                ncols = y-filter[1]+1
                shp = r,filter[0],filter[1],nrows,ncols
                strd = sr,sx,sy,sx,sy
                conv_matrices =  np.lib.stride_tricks.as_strided(inputs, shape=shp, strides=strd)

                # calculate actual filter updates
                d_filters = np.einsum('rijkl,rfkl->fij', conv_matrices, d_matrix)
                
                d_biases = np.sum(d_matrix, axis = (0,2,3))
 
                ''' update weights for each batch '''
                for l in range(num_layers + 1):
                    self.weights[l] -= self.learning_rate * d_weights[num_layers - l] / self.batch_size
                    
                    # add momentum if defined
                    if prev_d_weights is not None and self.momentum_factor != 0:
                        self.weights[l] -= self.learning_rate * self.momentum_factor * prev_d_weights[num_layers - l] / self.batch_size
                
                # Note: We lose a lot of information due to the max pooling layer which means the updates will be smaller thanw e want.
                # As such, we boost the size of the updates by multiplying them with self.pooling_parameter.
                self.filters -= self.learning_rate * d_filters / self.batch_size * self.pooling_parameter
                self.biases -= self.learning_rate * d_biases / self.batch_size * self.pooling_parameter
                
                # add momentum if defined
                if prev_d_weights is not None and self.momentum_factor != 0:
                    self.filters -= self.learning_rate * self.momentum_factor * prev_d_filters / self.batch_size * self.pooling_parameter
                    self.biases -= self.learning_rate * self.momentum_factor * prev_d_biases / self.batch_size * self.pooling_parameter
                
                # save weight changes for momentum
                prev_d_weights = d_weights.copy()
                prev_d_filter = d_filters.copy()  
                prev_d_biases = d_biases.copy()  

            # decrease learning rate
            if self.learning_rate != self.min_learning_rate:
                self.learning_rate = self.learning_rate_decay * self.learning_rate
                
                if self.learning_rate < self.min_learning_rate:
                    self.learning_rate = self.min_learning_rate
            
            # Print total computation time of current epoch
            print("Time: ", time.time() - start)
            
            # Track the training error rate
            #y_pred = self.predict(X_test)
            #error_rate = 1 - metrics.accuracy_score(y_test, y_pred)
            #print("Test Error rate: ", error_rate)


                    
    def initialize_weights(self, num_inputs, num_outputs):
        """Initializes the weight matrices
        :param num_inputs: int that represents the number of input nodes
        :param num_outputs: int that represents the number of output nodes
        """
        
        self.weights = [] # initialize list in which we store the weights
        
        np.random.seed(self.seed) # set the random generator seed
        weights = [] # list in which we temporarily store the weights
        
        layers = self.hidden_layers # the amount of units in each layer
        num_layers = len(layers) # amount of layers
        
        ''' initialize the weights according to a gaussian distribution. Every layer is scaled by a factor including the layer size such
        that we can prevent exploding weights. The bias weights are initialized to zero. '''
        
        weights.append(np.random.normal(loc=0.0, scale=1.0, size=(num_inputs, layers[0]) )  / np.sqrt(num_inputs / 2) )
        for l in range(num_layers - 1):
            weights.append(np.random.normal(loc=0.0, scale=1.0, size=(layers[l], layers[l+1]))  / np.sqrt(layers[l] / 2) )
        weights.append(np.random.normal(loc=0.0, scale=1.0, size=(layers[num_layers - 1], num_outputs) )  / np.sqrt(layers[num_layers - 1] / 2) )
        
        # add bias weights initialized to zero.
        for i in range(len(weights)):
            weights[i] = np.insert(weights[i], 0, 0, axis = 0)
        
        self.weights = weights # set weights
        
    def initialize_filters(self):
        """Initializes the filter weight matrices"""
        
        self.filters = []  # initialize list in which we store the filter weights

        filters = []  # list in which we temporarily store the lists

        num_filters = self.num_filter_layers  # amount of filters

        ''' initialize the filter weights according to a gaussian distribution. Every filters scaled by a factor including the layer size such
        that we can prevent exploding weights. The bias weights are initialized to zero. '''
        filters = np.random.normal(loc=0.0, scale=1.0, size=(num_filters, self.filter_shape[0], self.filter_shape[1])) / np.sqrt(self.filter_shape[0] * self.filter_shape[1] / 2)
        
        self.filters = filters  # set filters
        
        # initialize biases to zero
        self.biases = np.zeros(self.num_filter_layers)
        
    def normalize_training_data(self, X_data):
        """Normalizes the training data features to a range of [0, 1]
        :param X_data: NumPy matrix of input records. Each row represents one record and the columns represent features.
        :return: The normalized data in a NumPy array
        """
        
        new_data = X_data.copy()
        
        features_min = np.amin(new_data, 0) # lowest value found in the training data for each feature
        features_max = np.amax(new_data, 0) # highest value found in the training data for each feature
        max_min_diffs = (features_max - features_min)
        max_min_diffs[max_min_diffs == 0] += 1 # prevent dividing by zero
        
        # normalize data
        new_data = np.transpose( np.transpose(new_data - features_min) / max_min_diffs[:, None] )
        
        # save feature characteristics for use in prediction
        self.features_min = features_min
        self.max_min_diffs = max_min_diffs
        
        return new_data
    
    def forward_pass(self, records):
        """Performs a forward pass on a batch of records
        :param records: NumPy matrix of input records. Each row represents one record and the columns represent features.
                        The amount of rows is equal to batch_size
        :return: The activations for each layer for each record in a list of NumPy matrices, and the convolution values 
                 in a list of NumPy matrices
        """

        activations = [] # list in which we save all layer activations excluding the convolution layer
        
        # do a forward pass through the convolutional layers
        convolutions = self.convolutional_forward(records)
        
        # reshape the output of the convolutional layers to become one large input vector for the neural net
        inputs = convolutions[len(convolutions) - 1]
        records = inputs.reshape(records.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
        
        records = np.insert(records, 0, np.ones(records.shape[0]), axis = 1) # add bias constant to data
        
        # perform dropout
        dropout_array = np.random.choice([0, 1], size=records.shape, p=[self.deactivation_prob[0], 1 - self.deactivation_prob[0]])
        activations.append(records * dropout_array)
        
        layer = 0 # track layer index
        
        # pass records through network
        for w in self.weights:
            
            # calculate activation for current layer
            activation = np.dot(activations[layer], w)
            
            # ReLU activation function for all layers except output layer which has softmax:
            if layer != len(self.weights) - 1: # hidden layers
                
                # perform dropout
                dropout_array = np.random.choice([0, 1], size=activation.shape, p=[self.deactivation_prob[layer], 1 - self.deactivation_prob[layer]])
                activation = activation * dropout_array
                    
                activation[activation < 0] *= self.leaky_factor # ReLu function
                activation = np.insert(activation, 0, np.ones(activation.shape[0]), axis = 1) # insert bias
            else: # output layer
                max_activations = np.amax(activation, 1)
                exponentials = np.exp(activation - max_activations[:, None]) # calculate stable exponentials for softmax
                sum_totals = np.sum(exponentials, axis = 1) # calculate denominators
                activation = exponentials / sum_totals[:,None] # calculate softmax output
            
            activations.append(activation)
            layer += 1
        return activations, convolutions

    
    def convolutional_forward(self, X_data):
        """Performs a forward pass through the convolutional layers on a batch of records
        :param X_data: NumPy matrix of input records. Each row represents one record and the columns represent features.
        :return: List of outputs that contains: the input images, the convolution matrix, the matrix that keeps track
                 of which elements are the max in each pooling window, and the matrix containing the actual max values 
                 in each pooling window.
        """
        
        outputs = [] # list in which we store the outputs
        
        # WARNING: transforming the data type to 32 bit decreases the computation times, but it also decreases weight precision! 
        X_data = X_data.astype(dtype ='float32')   
        
        # reshape the input vector to an image matrix
        outputs.append(X_data.reshape(X_data.shape[0], int(math.sqrt(X_data.shape[1])), int(math.sqrt(X_data.shape[1]))))
        
        # Convolutions:
        output = self.convolution_kernel(outputs[0])
        outputs.append(output.copy())
        
        # ReLu function:
        output[output < 0] *= self.leaky_factor

        # Max pooling:
        output_activations, output_booleans = self.max_pooling(output)
        outputs.append(output_booleans)
        outputs.append(output_activations)

        return outputs

    def convolution_kernel(self, X_data):
        """Performs a forward pass through the convolutional layers on a batch of records
        :param X_data: NumPy matrix of input records. Each row represents one record and the columns represent features.
        :return: NumPy matrix containing the image matrices after convolutions
        """
        
        filter = self.filter_shape
        output = np.zeros( shape = (X_data.shape[0], self.num_filter_layers, X_data.shape[1] - filter[0] + 1, X_data.shape[2] - filter[1] + 1) )   
        
        # shift the kernel over the image and perform matrix multiplication
        for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                output[:, :, i, j] = np.einsum('rij,fij->rf', X_data[:, i:i+filter[0], j:j+filter[1]], self.filters) + self.biases
                
        return output
    
    def max_pooling(self, X_data):
        """Performs a forward pass through the convolutional layers on a batch of records
        :param X_data: NumPy matrix of input records. Each row represents one record and the columns represent features.
        :return: The matrix containing the actual max values in each pooling window, and the matrix that keeps track
                 of which elements are the max in each pooling window
        """
        
        stride = self.pooling_parameter
        
        # obtain all the submatrices from the images that need to be pooled
        r,f,x,y = X_data.shape
        sr, sf, sx, sy = X_data.strides    
        nrows = x-stride+1
        ncols = y-stride+1
        shp = r,f,stride,stride,nrows,ncols
        strd = sr,sf,sx,sy,sx,sy
        max_matrices =  np.lib.stride_tricks.as_strided(X_data, shape=shp, strides=strd)[:,:,:,:,::stride,::stride]
        
        # find max in each pool
        output_activations = np.amax( max_matrices, axis = (2, 3) )
        
        # keep track of which feature has the max value in each pool
        output_booleans = np.repeat(output_activations, stride, 2)
        output_booleans = np.repeat(output_booleans, stride, 3)
        
        output_booleans[X_data != output_booleans] = 0
        
        # ReLu function:
        output_booleans[output_booleans < 0] *= self.leaky_factor

        return output_activations, output_booleans
    
    
    def predict(self, X_data):
        """Predicts the labels of inputted records
        :param X_data: NumPy matrix of input records. Each row represents one record and the columns represent features.
        :return: The predicted labels in a NumPy array
        """
        
        X_data = np.transpose( (np.transpose(X_data - self.features_min) / self.max_min_diffs[:, None]) ) # normalize data
        
        convolutions = self.convolutional_forward(X_data) # forward pass throught the convolutional layers
        inputs = convolutions[len(convolutions) - 1]
        
        # reshape data matrices to become one input vector
        X_data = inputs.reshape(X_data.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
        
        X_data = np.insert(X_data, 0, np.ones(X_data.shape[0]), axis = 1) # add bias constant to data
        
        # For each record, find the highest output node activation and the corresponding prediction
        activation = X_data.copy()
        layer = 0 # track layer index
        
        for w in self.weights:
            activation = np.dot(activation, w)
                
            # ReLU activation function for all layers except output layer which has softmax:
            if layer != len(self.weights) - 1: # hidden layers
                activation[activation < 0] *= self.leaky_factor # ReLu function
                activation = np.insert(activation, 0, np.ones(activation.shape[0]), axis = 1) # insert bias
            else: # output layer
                max_activations = np.amax(activation, 1)
                exponentials = np.exp(activation - max_activations[:, None]) # calculate stable exponentials for softmax
                sum_totals = np.sum(exponentials, axis = 1) # calculate denominators
                activation = exponentials / sum_totals[:,None] # calculate softmax output
                    
            layer += 1
                    
        return np.argmax(activation, 1)
