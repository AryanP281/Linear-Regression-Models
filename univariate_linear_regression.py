#********************************Imports********************************
import random

#********************************Global Variables********************************
theta0 = 0.0 #The theta0 parameter
theta1 = 0.0 #The theta1 parameter
#Thus h(x) = theta0 + (theta1)*(x)
learning_rate = 0.001 #The learning rate for gradient descent
cost_func_value = 0.0 #The value of the cost function befor the last update of parameter values. Acts as an indicator of the model's accuracy

#********************************Functions********************************
def train(inputs, expected_outputs) :
    """Trains the linear regression model untill cost function is minimized"""

    #The 1st iteration
    train_batch(inputs, expected_outputs)
    last_cost_func_value = cost_func_value #The value of the cost function during the last iteration

    #Training untill cost function is minimized
    change_in_cost_func = -1
    inflection_point = cost_func_value
    inflection_point_found = False
    while(True) :
        if(inflection_point_found) :
            if(cost_func_value - inflection_point > (inflection_point * 5) / 100) :
                break

        train_batch(inputs, expected_outputs)
        change_in_cost_func = cost_func_value - last_cost_func_value
        last_cost_func_value = cost_func_value
        if(change_in_cost_func > 0) :
            if(not inflection_point_found) :
                inflection_point = cost_func_value
                inflection_point_found = True
        else :
            inflection_point = 0.0
            inflection_point_found = False

    print("Training complete")


def train_for_epochs(inputs, expected_outputs, epochs) :
    """Trains the linear regression model for the given number of epochs"""
    
    #Training for the given number of epochs
    for epoch in range(0, epochs) :
       train_batch(inputs, expected_outputs)

    print("Training complete")

def train_batch(inputs, expected_outputs) :
    """Runs the training algorithm"""

    #Loading the global variables
    global cost_func_value
    cost_func_value = 0.0 #Resetting
    dataset_size = len(inputs)

    derivatives = [0,0] #The derivatives i.e [dJ/dtheta0, dJ/dtheta1] for the current iteration

    #Iterating through the dataset
    for a in range(0, dataset_size) :
        #Calculating the error
        error = h(inputs[a]) - expected_outputs[a]

        #Updating the derivatives
        derivatives[0] += error
        derivatives[1] += error * inputs[a]

        #Updating the cost function value
        cost_func_value += error ** 2

    #Calculating the final derivatives after iterating through the dataset
    derivatives[0] /= dataset_size
    derivatives[1] /= dataset_size

    #Calculating the cost function value
    cost_func_value /= 2 * dataset_size

    #Updating parameters using Gradient Descent
    gradient_descent(derivatives)

def h(x) :
    """Returns h(x) = theta0 + (theta1)*(x)"""

    return theta0 + (theta1 * x)

def gradient_descent(derivatives) :
    """Updates the parameters i.e theta0 and theta1 using gradient descent.
    dJ/d(theta0) = (error_1 + error_2 + ... + error_n) / len(dataset)
    dJ/d(theta1) = (error_1*x_1 + error_2*x_2 + ... + error_n*x_n) / len(dataset)"""

    #Loading global variables
    global theta0, theta1

    #Updating theta0
    theta0 -= learning_rate * derivatives[0]

    #Updating theta1
    theta1 -= learning_rate * derivatives[1]

def set_learning_rate(new_learning_rate)  :
    """Changes the learning rate"""

    global learning_rate
    learning_rate = new_learning_rate

def get_cost_function_value() :
    """Returns the value of the cost function"""

    return cost_func_value

def reset() :
    """Resets the model parameters"""

    global theta0, theta1, learning_rate, cost_func_value

    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.001
    cost_func_value = 0.0