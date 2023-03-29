import pandas as pd
import numpy as np

#takes in a dataframe of features
#returns a new dataframe of features scaled using mean normalization
def mean_normalize(X):
    
    scaled_X = {}
    #calculates mean and variance of each feature then scales it
    for feature in X.columns:
        
        #calculates mean
        mean = 0
        for sample_weight in X[feature]:
            mean += sample_weight
        mean /= len(X)

        #calculates population variance
        pop_variance = 0
        for sample_weight in X[feature]:
            pop_variance += (sample_weight - mean)**2
        pop_variance /= (len(X))

        #scales weight and updates values in the dataframe, updates feature column name, and updates variable holding feature column name
        row = []
        for i in range(len(X)):
            row.append((X[feature][i] - mean) / (pop_variance**0.5))
        scaled_X['scaled_' + feature] = row
        
    scaled_X = pd.DataFrame(scaled_X)
    return scaled_X

#takes in dataframe of features, a dataframe of the target variable, and a learning rate and convergence constant optionally
#returns a list of parameters
def gradient_descent(X, y, learning_rate=0.2, convergence_constant=0.00001):
    
    #adds feature x0 to feature matrix which is a vector of 1's
    x0 = []
    for i in range(len(X)):
        x0.append(1)
    X.insert(0, 'x0', x0)
    
    #initializes parameters to zero
    num_params = len(X.columns)
    old_params = []
    temp_params = []
    new_params = []
    for i in range(num_params):
        old_params.append(0)
        temp_params.append(0)
        new_params.append(0)
        
    #creates target and list of features
    target = y.columns[0]
    features = []
    for feature in X.columns:
        features.append(feature)
  
    converged = False
    while converged != True:
        
        #makes the parameters learned from the previous iteration the old parameters
        for i in range(num_params):
            old_params[i] = new_params[i]
        
        #calculates new regression coefficients
        for i in range(len(features)):
            derivative = 0 
            temp = 0
            for j in range(len(X)):
                for k in range(len(features)):
                    temp += old_params[k] * X[features[k]][j]
                temp -= y[target][j]
                temp *= X[features[i]][j]
                temp /= len(X)
                derivative += temp
            temp_params[i] = old_params[i] - (learning_rate * derivative)
            
        #updates parameters with new values simulataneously
        for i in range(num_params):
            new_params[i] = temp_params[i]
         
        #determines whether descent has converged
        converged = True
        for i in range(num_params):
            if abs(new_params[i] - old_params[i]) >= convergence_constant:
                converged = False
       
    #removes x0
    X.drop(['x0'], axis=1, inplace=True)
    
    return new_params

#takes in dataframe of features and a dataframe of the target variable
#returns a list of parameters
def normal_equations(X, y):
    
    #adds feature x0 to feature matrix which is a vector of 1's
    x0 = []
    for i in range(len(X)):
        x0.append(1)
    X.insert(0, 'x0', x0)
    
    #creates X^T matrix
    X_tran = X.transpose()

    #performs matrix multiplication X^T * X
    term1 = X_tran.dot(X)

    #inverts product of X^T * X
    matrix1 = np.linalg.inv(term1)

    #performs matrix multiplication X^T * y
    matrix2 = X_tran.dot(y)

    #performs matrix multiplication of matrix1 (X^T * X)^-1 and matrix2 (X^T * y)
    params = matrix1.dot(matrix2)
    
    #removes x0
    X.drop(['x0'], axis=1, inplace=True)
    
    return params