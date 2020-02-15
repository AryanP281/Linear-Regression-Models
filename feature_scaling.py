"""Provides functions for feature scaling"""

#********************************Imports********************************

#********************************Functions********************************
def min_max_normalization_multivariate(dataset) :
    """Scales the multivariate data using min-max normalization"""

    scaled_data = [] #The data obtained afte scaling the given dataset
    
    #Getting the max and min values of the features
    max_vals = []
    min_vals = []
    for feature_index in range(0, len(dataset[0])) :
        min_vals.append(get_min_val_of_feature(dataset, feature_index))
        max_vals.append(get_max_val_of_feature(dataset, feature_index))

    #Calculating the scaled features
    for a in range(0, len(dataset)) :
        scaled_data.append([])
        for feature_index in range(0, len(dataset[a])) :
            scaled_val = (dataset[a][feature_index] - min_vals[feature_index]) / (max_vals[feature_index] - min_vals[feature_index])
            scaled_data[-1].append(scaled_val)
            
    #Returning the scaled 
    return scaled_data

def min_max_normalization_univariate(data) :
    """Scales the univariate data using min-max normalization"""

    scaled_data = [] #The data obtained afte scaling the given data

    min_val = min(data) #The min value of the feature 
    max_val = max(data) #The max value of the feature

    #Calculating the scaled features
    for a in range(0, len(data)) :
        try :
            scaled_val = (data[a] - min_val) / (max_val - min_val)
        except ZeroDivisionError :
            scaled_val = 0
            print("Divide by zero encountered. Scaled feature set to 0")
        scaled_data.append(scaled_val)

    return scaled_data
        

#********************************Helper Functions********************************
def get_min_val_of_feature(dataset, feature_index) :
    """Gets the min value of the given feature from the dataset"""

    feature = [] #Value of the feature in the dataset
    for inputs in dataset :
        feature.append(inputs[feature_index])

    #Returning the min val
    return min(feature)

def get_max_val_of_feature(dataset, feature_index) :
    """Gets the max value of the given feature from the dataset"""

    feature = [] #Value of the feature in the dataset
    for inputs in dataset :
        feature.append(inputs[feature_index])

    #Returning the max val
    return max(feature)