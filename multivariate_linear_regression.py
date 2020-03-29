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

        X = np.concatenate((np.ones((len(training_inputs), 1)), training_inputs), axis=1) #Converting the training inputs to NumPy array

        y = np.array(expected_outputs) #Coverting the expected outputs to NumPy array
        y.shape = (len(expected_outputs), 1) #Reshaping the expected outputs vector

        #Checking if the user wants epoch based training
        if(epochs != None) :
            #Training for the given number of epochs
            for a in range(0, epochs) :
                derivatives = np.zeros(self.parameters.shape)

                #Getting the outputs
                forwardprop_res = self.forwardpropogate(X,y,self.parameters,lmbda) # (Z, E, J)
                self.cost_func_value = forwardprop_res[2]
                E = forwardprop_res[1]

                #Getting the gradients
                derivatives = self.get_gradients(self.parameters, X, y, E, lmbda)

                #Using gradient descent for optimizing the model parameters
                self.gradient_descent(derivatives, learning_rate)

    def forwardpropogate(self, X, y, parameters, lmbda) :
        """Forward propogates and returns the outputs and cost function value for the given inputs.
        X = inputs
        y = expected outputs
        parameters = parameters to be used
        lmbda = Regularization Parameter
        returns (Outputs, Errors, Cost)"""

        #The model output using the given parameters
        Z = np.dot(X, parameters)

        #Calculating the cost function value
        E = Z - y
        j = np.sum(E**2) / (2 * y.shape[0])
        j += (lmbda / (2 * y.shape[0])) * np.sum((parameters[1:-1] ** 2))

        #Returning the outputs and cost
        return (Z, E, j)

    def get_gradients(self, params, X, y, E, lmbda) :
        """Calculates and returns the gradients for the parameters
        params = the parameters whose gradients are required,
        X = inputs,
        y = expected outputs,
        E = errors in the model's outputs i.e (h - y),
        lmbda = Regularization Parameter
        return grads = a Numpy array containing the calculated gradients"""

        grads = np.zeros(params.shape) #The gradients

        #Calculating the gradients
        grads = np.dot(X.T, E) / y.shape[0]  
        grads[1:] += params[1:] * (lmbda / y.shape[0])

        #Returning the calculated gradients
        return grads


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

    def predict(self, inputs) :
        """Gets prediction from the model"""

        #Converting the inputs to the required format
        x = np.array([self.x0] + inputs)

        #Calculating the output
        output = self.h(x)

        return output

