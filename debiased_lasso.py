''' debiased LASSO '''
''' from the lasso we select 4 features
    constant, sqrt_living, view, waterfront
    this leaves us with 1.23e15
    now lets do a least sq
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
os.system('cls')

####load data and some data manupulation
# load data,train data and test data
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float,
  'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
  'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str,
   'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

data = pd.read_csv('kc_house_data.csv',dtype = dtype_dict)
data_train = pd.read_csv('kc_house_train_data.csv',dtype = dtype_dict)
data_test = pd.read_csv('kc_house_test_data.csv',dtype = dtype_dict)

def to_numpy(df, fetures_list, target):
    '''a function that transforms dataframe in pandas to numpy array
    Input:
    1.Data frame(pandas data frame format),2.features(list),3.target(string)
    Output:
    H(numpy array) and  Y(numpy array)
    '''
    df['constant'] = 1
    features_list = ['constant'] + fetures_list
    feature_df = df[features_list]
    feature_matrix = feature_df.as_matrix()
    y_series=df[target]
    output_array = y_series.as_matrix()
    return ( feature_matrix, output_array )


def normalize_feature(feature_matrix):
    '''
    function that normalize features_list
    INPUT:
    matrix of feature(numpy array)
    OUTPUT:
    1.normalized matrix of feature( numpy array)
    2. norms(numpy array) of features which wil be used for prediction of test
    data
    '''
    norms = np.linalg.norm(feature_matrix, axis=0)
    feature_matrix_normalized = feature_matrix / norms
    return (feature_matrix_normalized,norms)

# main loop

def ls_coordinate(feature_matrix, output_array, tolerence, initial_weights):
    ''' instead of gradient descent we use coordinate descent for LS
    OUTPUT : LS weights
    '''

    # initial variables
    weights = np.array(initial_weights)
    converged = 0
    counter = 0
    # number of features
    D = feature_matrix.shape[1]
    while converged ==0 :
        counter = counter + 1
        weights_old = np.array([i for i in weights])
        # for each cordinate we solve ls while other cordinates(j) are fixed
        for j in range(D):
            # predict output with current weights
            prediction = np.dot(feature_matrix,weights)
            # comput the error between predicted output and real outputs
            error = output_array - prediction
            # comput error if feature j is not considered in predicting yhat
            error_without_j = error + (weights[j] * feature_matrix[:,j])
            # compute ro+j which is the correletion between feature j an
            # error if feature j is not considered
            ro_j = np.dot(feature_matrix[:,j] , error_without_j)
            # for all cordinated except constant feature we shrink weights
            weights[j] = ro_j

        # stoping condition
        #after each cyle over all cordinate we look for the maximum change in
        # cordination. if it is too small we stop.
        weights_change = weights - weights_old
        t = np.amax(np.absolute(weights_change))
        if t < tolerence :
            converged = 1

    return (weights)

selected_features = [ 'sqft_living', 'view', 'waterfront']

(feature_matrix_train , output_train) = to_numpy(data_train, selected_features , 'price')
# normalize them
(feature_matrix_n_train,norms_train) = normalize_feature(feature_matrix_train)
# convert test data in csv format to numpy
(feature_matrix_test , output_test) = to_numpy(data_test, selected_features , 'price')

tolerence = 1
num_features = len(selected_features) + 1
initial_weights = np.zeros(num_features)
weights_ls = ls_coordinate(feature_matrix_n_train, output_train,tolerence, \
initial_weights)

print "w_LS"
print weights_ls
print " weights for24,48,3.3,7.3"

weights_normalized = np.divide(np.array(weights_ls,dtype=float),\
np.array(norms_train,dtype=float))

prediction_train = np.dot(feature_matrix_n_train, weights_ls)
rss_train = np.dot((output_train - prediction_train),\
(output_train - prediction_train))
print "RSS of train data in LS %s:" %rss_train
print "RSS for train data with these features in lasso was 1.23e15"
print "RSS for train data with 13 feature is .83e14"

prediction_test = np.dot(feature_matrix_test, weights_normalized)
# risidual sum of square with optimized w for test data
rss_test = np.dot((output_test - prediction_test),\
(output_test - prediction_test))
print "RSS of test data in LS %s:" %rss_test
print "RSS for test data with these features in lasso was 2.7e14"
print "RSS for test data with 13 feature is 1.9e14"
