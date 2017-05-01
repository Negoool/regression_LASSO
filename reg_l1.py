""" Lasso """

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
# print the basic information about the data
#data.info()
#print data.head()
print " \n number of train data points : %d " %(data_train.shape[0])
print " number of test data points : %d " %(data_test.shape[0])

# some plotting
# data.plot( x = 'sqft_living' , y = 'price', kind = 'scatter')
# plt.figure
# data.plot( x = 'yr_renovated' , y = 'price', kind = 'scatter', xlim = [1950,2015])
# plt.figure
# data.plot( x = 'bedrooms' , y = 'price', kind = 'scatter')
# plt.show()


# defining new features
# define square # of bedrooms to distinguish between a house with
# ... small number of bedrooms and a house with lots of betrooms
data['bedroom_square'] = data['bedrooms'] * data['bedrooms']
# define sqrt of squarefeet as another feature to pale the role of sqft
data['sqft_living_sqrt'] = np.sqrt(data['sqft_living'])
data['sqft_lot_sqrt'] = np.sqrt(data['sqft_lot'])

#### Lasso

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

def lasso_cordinate(feature_matrix, output_array, landa, tolerence, \
initial_weights):
    ''' solve lasso with coordinate descent algorithm
    INPUT: feature_matrix(numpy array) and it has to be normalized
       output_array(numpy array)
       landa(int variable) l1 penalty which balances RSS(obj) and l1 norm
       tolerence(int/float variable) which determines stoping condition
       intial_weights(list/numpy array),with the same size as number of features
    OUTPUT: optimized weights(numpy array) for RSS + l1*norm1
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
        # for each cordinate we solve lasso while other cordinates(j) are fixed
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
            if j != 0 :
                if ro_j < (-landa/2):
                    weights[j] = ro_j + (landa/2)
                elif ro_j > (landa/2):
                    weights[j] = ro_j - (landa/2)
                else:
                    # ro_j is between those numbers which is quite small
                    weights[j] = 0
            else :
                weights [j] = ro_j

        # stoping condition
        #after each cyle over all cordinate we look for the maximum change in
        # cordination. if it is too small we stop.
        weights_change = weights - weights_old
        t = np.amax(np.absolute(weights_change))
        if t < tolerence :
            converged = 1

    return (weights)

### train my lasso
#a sunset of all features which is considered for this regression
my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                'waterfront', 'view', 'condition', 'grade','sqft_above',
                'sqft_basement','yr_built','yr_renovated']

# convert train data in csv format to numpy
(feature_matrix_train , output_train) = to_numpy(data_train, my_features , 'price')
# normalize them
(feature_matrix_n_train,norms_train) = normalize_feature(feature_matrix_train)
# convert test data in csv format to numpy
(feature_matrix_test , output_test) = to_numpy(data_test, my_features , 'price')
# normalize them,
#************* there is no need for this *********
# feature_matrix_n_test = normalize_feature(feature_matrix_test)

# number of landas
num_landas = 6
# creat a set of landas in logspace
landa = np.logspace(5, 8, num_landas)
# num_features, we add 1 for constant feature
num_features = len(my_features) + 1
# a matrix of all weights, rss for train and test data for different landas
all_weights = np.zeros((num_landas,num_features))
RMSE_train = np.zeros(num_landas)
RSS_train = np.zeros(num_landas)
RSS_test = np.zeros(num_landas)
RMSE_test = np.zeros(num_landas)
nnz = np.zeros(num_landas)
# inputs of function
tolerence = 100
initial_weights = np.zeros(num_features)


## lasso for couple of landas
for i in range(num_landas):
    weights = lasso_cordinate(feature_matrix_n_train, output_train,landa[i],\
    tolerence,initial_weights)
    #save weights
    all_weights[i,:] = weights
    # for using w for test data we either have to normalize the data test as we
    # did train data or we have to normalize weights
    weights_normalized = np.divide(np.array(weights,dtype=float),\
     np.array(norms_train,dtype=float))
    # count number of non zero elemnets of weights
    nnz[i] = np.count_nonzero(weights, axis=1)
    # predicted output for train data with optimized w
    prediction_train = np.dot(feature_matrix_n_train, weights)
    # risidual sum of suare with optimized w for train data
    rss_train_landa = np.dot((output_train - prediction_train),\
    (output_train - prediction_train))
    # save rss as the obj function
    RSS_train[i] = rss_train_landa
    # for the sake of comparison, convert rss to rmse
    RMSE_train_landa = np.sqrt(rss_train_landa / len(output_train))
    # save rmse  of this landa
    RMSE_train[i] = RMSE_train_landa
    # predicted output for test data with optimized w
    prediction_test = np.dot(feature_matrix_test, weights_normalized)
    # risidual sum of square with optimized w for test data
    rss_test_landa = np.dot((output_test - prediction_test),\
    (output_test - prediction_test))
    # save RSS test data
    RSS_test[i] = rss_test_landa
    # for the sake of comparison, convert rss to rmse
    RMSE_test_landa = np.sqrt(rss_test_landa / len(output_test))
    # save rmse of this landa
    RMSE_test[i] = RMSE_test_landa


print RSS_test
print RSS_train
# print result of this
print "-"*70
print "landa"+" "*14+"RSS_train"+" "*7+"nnz"+" "*7+"RMSE_train"+" "*10+"RMSE_test"
print "-"*70
for i in range(num_landas):
    print "%14s | %15s | %7s |%14s|%14s" %(landa[i],RSS_train[i] , nnz[i], \
    RMSE_train[i], RMSE_test[i])
print "-" *70


my_features = ['constant'] + my_features
for i in range(num_landas):
    if i==0:
        print " "*13,
    print "%12s" %landa[i],

print "-"*75
for j in range(len(my_features)):
    print
    for i in range(num_landas):
        if i ==0:
            print "%15s" %(my_features[j]),
        print"%11.1f" %(all_weights[i,j]),




# plot rmse of train and test data for values of l1_penalty
plt.subplot(2, 1, 1)
plt.semilogx(landa, RSS_train, 'ro', label = 'training')
plt.semilogx(landa, RSS_test, 'bo', label = 'test')
plt.ylabel('RMSE')
plt.title('RMSE of train and test data VS l1_penalty')
plt.legend()

# plot number of non zero features for values of l1_penalty
plt.subplot(2, 1, 2)
plt.semilogx(landa , nnz , 'gs')
plt.ylabel('number of non zero features')
plt.xlabel('l1_penalty')
plt.title('number of non zero features VS l1_penalty')
plt.show()
