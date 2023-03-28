import numpy as np
import warnings

#takes in a dataframe of features
#scales each feature using mean normalization
def scale_features(X):
    
    warnings.filterwarnings('ignore')
    scaled_X = X
    #calculates mean and variance of each feature then scales it
    for feature in X.columns:
        
        #calculates mean
        mean = 0
        for sample_weight in scaled_X[feature]:
            mean += sample_weight
        mean /= len(scaled_X)

        #calculates population variance
        pop_variance = 0
        for sample_weight in scaled_X[feature]:
            pop_variance += (sample_weight - mean)**2
        pop_variance /= (len(scaled_X))

        #scales weight and updates values in the dataframe, updates feature column name, and updates variable holding feature column name
        for i in range(len(scaled_X)):
            scaled_X[feature][i] = (X[feature][i] - mean) / (pop_variance**0.5)
        scaled_X.rename(columns = {feature :'scaled_' + feature}, inplace=True)
        
    return scaled_X

#takes in dataframe of features, a dataframe of the target variable, and a learning rate and convergence constant optionally
#returns a list of parameters
def gradient_descent(X, y, learning_rate=0.2, convergence_constant=0.00001):
    
    #scales features
    X = scale_features(X)
    
    #initializes parameters to zero
    num_params = len(X.columns) + 1
    old_params = []
    temp_params = []
    new_params = []
    for i in range(num_params):
        old_params.append(0)
        temp_params.append(0)
        new_params.append(0)
        
    #creates list of features
    features = []
    for feature in X.columns:
        features.append(feature)
  
    converged = False
    while converged != True:
        
        #makes the parameters learned from the previous iteration the old parameters
        for i in range(num_params):
            old_params[i] = new_params[i]
        
        #calculates new bias
        derivative = 0
        for i in range(len(X)):
            derivative += old_params[0]
            for j in range(len(features)):
                derivative += (old_params[j + 1] * X[features[j]][i])
            derivative -= y[y.columns[0]][i]
        derivative /= len(X)
        temp_params[0] = old_params[0] - (learning_rate * derivative)
        
        #calculates new regression coefficients
        for i in range(len(features)):
            derivative = 0
            temp = 0
            for j in range(len(X)):
                temp += old_params[0]
                for k in range(len(features)):
                    temp += (old_params[k + 1] * X[features[k]][j])
                temp -= y[y.columns[0]][j]
                temp *= X[features[i]][j]
                temp /= len(X)
                derivative += temp
            temp_params[i + 1] = old_params[i + 1] - (learning_rate * derivative)
            
        #updates parameters with new values simulataneously
        for i in range(num_params):
            new_params[i] = temp_params[i]
         
        #determines whether descent has converged
        converged = True
        for i in range(num_params):
            if abs(new_params[i] - old_params[i]) >= convergence_constant:
                converged = False
                
    return new_params

#takes in dataframe of features and a dataframe of the target variable
#returns a list of parameters
def normal_equations(X, y):
    
    #scales features
    X = scale_features(X)
    
    #adds column of 1's to represent x0 which is always one
    vector = []
    for i in range(len(X)):
        vector.append(1)
    X.insert(0, 'x0', vector)
    
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
    
    return params