import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize matrix weights (Refer to must read article "http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf")
        # Weights should be chosen randomly but in such a way that the sigmoid is primarily activated in its linear region! Recall that Sigmoid activation function to be used is in fact integral of a normal distribution. To obtain a satndard deviation close to 1 at the output of the first hidden layer,the input to the sigmoid should also have a standard deviation of 1. For that to happen, weights should be randomly drawn from a distribution with mean zero and standard deviation (scale) of 1 / sqrt(m) where m is the number of inputs to the unit. 
        self.weights_input_to_hidden = np.random.normal(loc=0.0, scale=self.input_nodes**-0.5, 
                                       size=(self.input_nodes, self.hidden_nodes)) # tuple size of weight matrix 

        self.weights_hidden_to_output = np.random.normal(loc=0.0, scale=self.hidden_nodes**-0.5, 
                                       size=(self.hidden_nodes, self.output_nodes)) # tuple size of weight matrix
        self.lr = learning_rate
              
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        
        self.activation_function = lambda x : 1/(1+np.exp(-x)) 
                    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(shape=self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(shape=self.weights_hidden_to_output.shape)
        # The zip() function returns a zip object, which is an iterator of tuples where the first items in each passed iterator are paired together, and then the second items in each passed iterator are paired together etc. Take important note that arguments should be numpy array not dataframes! Dataframes should be converted to arrays using .values.
        for X, y in zip(features, targets):  
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        # Update weights
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X[None, :], self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        # Despite hidden layer, sigmoid activation function is replaced with activation function of f(x)=x as we are not trying to get a probablity but a real number (count of rides). So in this case the final output will be equal to final input.
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        
        output_error_term = error * 1 # Since we have sigmoid(x) = x at output layer; derivative is 1
        
        # TODO: Calculate the hidden layer's contribution to the error
         
        hidden_error = error * self.weights_hidden_to_output
        # derivative of hidden layer activation function ddxσ(x)=σ(x)(1−σ(x)) 
        hidden_error_term = hidden_error.T * (hidden_outputs * (1.0 - hidden_outputs))
        
        # Weight step (input to hidden)
        delta_weights_i_h += np.dot(X[:, None], hidden_error_term)
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs.T

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr / n_records * delta_weights_h_o 
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr / n_records * delta_weights_i_h  

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 20000
learning_rate = 0.1
hidden_nodes = 8
output_nodes = 1