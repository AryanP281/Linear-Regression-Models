#********************************Imports********************************
import numpy as np

#********************************Classes********************************
class Multivariate_Linear_Regression_Model(object) :

    x0 = 1.0 #The value of the bias

    def __init__(self, features_count = 1) :

        #Initializing the models parameters
        self.parameters = np.zeros(features_count + 1) #A NumPy array containing the model's parameters
        self.parameters.shape = (features_count + 1, 1)

        self.cost_func_value = 0.0 #The last calculated value of the cost function

    def train_with_gradient_descent(self, training_inputs, expected_outputs, learning_rate = 0.001, epochs = None, use_regularization=False, lmbda=0) :
        """Trains the model on the given data for the given number of epochs"""

        input_array = self.generate_input_array(training_inputs) #Converting the training inputs to NumPy array
        expected_outputs_array = np.array(expected_outputs) #Coverting the expected outputs to NumPy array

        training_dataset_size = len(training_inputs) #The size of the training data

        #Checking if the user wants epoch based training
        if(epochs != None) :
            #Training for the given number of epochs
            for a in range(0, epochs) :
                derivatives = np.zeros(self.parameters.size) #The derivatives of the cost function wrt the parameters
                #Iterating through the given data
                for b in range(0, training_dataset_size) :
                    output = self.h(input_array[b])
                    error = output - expected_outputs_array[b] #The error in the model's prediction

                    #Updating the derivatives
                    for c in range(0, derivatives.size) :
                        derivatives[c] += error[0] * input_array[b][c]
                    
                    #Updating the cost function value
                    self.cost_func_value += float(error[0] ** 2)

                #Calculating the final derivatives
                derivatives = derivatives / input_array.size

                #Calculating the final value of the cost function
                self.cost_func_value /= 2 * input_array.size

                #Checking if regularization is to be used
                if(use_regularization) :
                    params_sq_sum = 0 #The sum of the squares of the parameters (required to calculate cost)
                    for param in range(1, self.parameters.shape[0]) :
                        derivatives[param][0] += lmbda * self.parameters[param][0] / input_array.size
                        params_sq_sum += self.parameters[param][0]**2
                    
                    #Calculating the final value of the cost function
                    self.cost_func_value += lmbda * params_sq_sum

                #Using gradient descent for optimizing the model parameters
                self.gradient_descent(derivatives, learning_rate)

    def train_with_vectorized_gradient_descent(self, training_inputs, expected_outputs, learning_rate = 0.001, epochs = None, use_regularization=False, lmbda=0) :
        """Trains the model on the given data for the given number of epochs"""

        X = self.generate_input_array(training_inputs) #Converting the training inputs to NumPy array

        y = np.array(expected_outputs) #Coverting the expected outputs to NumPy array
        y.shape = (len(expected_outputs), 1) #Reshaping the expected outputs vector

        #Checking if the user wants epoch based training
        if(epochs != None) :
            #Training for the given number of epochs
            for a in range(0, epochs) :
                derivatives = np.zeros(self.parameters.shape)
                Z = np.dot(X, self.parameters)
                E = Z - y

                #Calculating the cost function
                self.cost_func_value = sum(E**2) / (2 * y.shape[0])

                #Calculating the gradients
                derivatives = np.dot(X.T, E) / y.shape[0]

                #Checking if regularization is to be used
                if(use_regularization) :
                    derivatives[1:-1] += self.parameters[1:-1] * (lmbda / y.shape[0])
                    self.cost_func_value += (lmbda / (2 * y.shape[0])) * (self.parameters[1:-1] ** 2)

                #Using gradient descent for optimizing the model parameters
                self.gradient_descent(derivatives, learning_rate)

    def h(self, inputs) :
        """Calculates the output of the model i.e y = theta0*x_0 + theta1*x_1 + .... + theta_n*x_n = parameters.T * inputs"""

        inputs.shape = (inputs.size, 1) #Setting the inputs to the required shape
        output_vector = np.dot(self.parameters.T, inputs)

        return output_vector

    def gradient_descent(self, derivatives, learning_rate) :
        """Updates the parameters of the model using gradient descent"""

        for a in range(0, self.parameters.size) :
            self.parameters[a] -= learning_rate * derivatives[a]

    def train_with_normal_equation(self, training_inputs, expected_outputs) :
        """Trains the model using the Normal Equation method"""

        X = self.generate_input_array(training_inputs) #Converting the training inputs to NumPy array
        Y = np.array(expected_outputs) #Coverting the expected outputs to NumPy array

        #According to the Normal Equation, THETA = ((X.T*X)^-1) * X.T * Y
        self.parameters = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

    def generate_input_array(self, inputs) :
        """"Creates and returns a NumPy array containing the x0 and the inputs"""

        X = np.array([self.x0] + inputs[0]) #Adding the 1st inputs to set the array dimensions
        for a in range(1, len(inputs)) :
            X = np.vstack((X, [self.x0] + inputs[a]))

        X.shape = (len(inputs), len(inputs[0]) + 1) #Setting the dimensions(shape) of the input matrix
        return X

    def reset(self) :
        """Resets all the model parameters to default values"""

        self.parameters = np.zeros(self.parameters.size)
        self.parameters.shape = (self.parameters.size, 1)

        self.cost_func_value = 0.0

    def get_output(self, inputs) :
        """Gets output from the model"""

        #Converting the inputs to the required format
        x = np.array([self.x0] + inputs)

        #Calculating the output
        output = self.h(x)

        return output

