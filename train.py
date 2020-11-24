import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import sample

def create_gaussian_kernel_function(gamma):
    '''
    This function creates and returns a separate function that uses the given gamma value as its gamma hyperparameter. This is necessary since the gaussian kernel function accepted by the KNeighborsRegressor can only accept a single parameters: an array of distances. In order to dynamically configure the gammas, I have to dewfine a new function at run-time that uses that gamma in its base definition.
    '''
    def gaussian_kernel_function(distances):
        weights = np.exp(-gamma * (distances**2))
        return weights / np.sum(weights)
    return gaussian_kernel_function

def create_input_array(documents, phrase_len, min_df, max_df):
    '''
    This function creates a bag of words (X / input) matrix to be used in the model based on the given documents
    '''
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(phrase_len, phrase_len), min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(documents)
    return X

def load_data(fold, phrase_len, min_df, max_df):
    #Gather and process data into a form that we can run ML on
    documents, y = sample.main()
    X = create_input_array(documents, phrase_len, min_df, max_df)

    #Split into train and test segments
    indices = list(np.arange(X.size))
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=1 / fold)
    return X, y, xtrain, xtest, ytrain, ytest

def evaluations(X, y, xtrain, xtest, ytrain, ytest, k, gamma, alpha, phrase_len):
    #Create models
    #knn_model = KNeighborsRegressor(n_neighbors=k, weights=create_gaussian_kernel_function(gamma)).fit(X[train], y[train])
    lin_reg_model = LinearRegression().fit(xtrain, ytrain)
    lasso_model = Lasso(alpha=alpha).fit(xtrain, ytrain)
    ridge_model = Ridge(alpha=alpha).fit(xtrain, ytrain)
    baseline = DummyRegressor(strategy="mean").fit(xtrain, ytrain)

    #Evaluate models
    #knn_mse = mean_squared_error(y[test], knn_model.predict(X[test]))
    lin_reg_mse = mean_squared_error(ytest, lin_reg_model.predict(xtest))
    lasso_mse = mean_squared_error(ytest, lasso_model.predict(xtest))
    ridge_mse = mean_squared_error(ytest, ridge_model.predict(xtest))
    baseline_mse = mean_squared_error(ytest, baseline.predict(xtest))

    return phrase_len,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse

def main():
    #evaluate models
    #(hyper)parameters
    phrase_len = 2
    min_df = int(1) #int for absolute counts, float for proportion
    max_df= float(1.0) #int for absolute counts, float for proportion
    gamma = 0
    k = 3
    alpha = .01
    fold = 5

    input_datas = []
    max_dfs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for val in max_dfs:
        input_datas.append(load_data(fold, phrase_len, min_df, val))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(X, y, xtrain, xtest, ytrain, ytest, k, gamma, alpha, max_dfs[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Model evaluations (MSE)')
    print('Data for max DF')
    print('max_df,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

    #evaluate models
    #(hyper)parameters
    phrase_len = 2
    min_df = int(1) #int for absolute counts, float for proportion
    max_df= float(1.0) #int for absolute counts, float for proportion
    gamma = 0
    k = 3
    alpha = .01
    fold = 5

    input_datas = []
    min_dfs = [int(1)]
    for val in min_dfs:
        input_datas.append(load_data(fold, phrase_len, val, max_df))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(X, y, xtrain, xtest, ytrain, ytest, k, gamma, alpha, min_dfs[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Model evaluations (MSE)')
    print('Data for min DF')
    print('min_df,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

    #evaluate models
    #(hyper)parameters
    phrase_len = 2
    min_df = int(1) #int for absolute counts, float for proportion
    max_df= float(1.0) #int for absolute counts, float for proportion
    gamma = 0
    k = 3
    alpha = .01
    fold = 5

    input_datas = []
    phrase_lens = [1, 3, 5, 7, 9]
    for val in phrase_lens:
        input_datas.append(load_data(fold, val, min_df, max_df))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(X, y, xtrain, xtest, ytrain, ytest, k, gamma, alpha, phrase_lens[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Model evaluations (MSE)')
    print('Data for phrase length')
    print('phrase_len,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

    #evaluate models
    #(hyper)parameters
    phrase_len = 2
    min_df = int(1) #int for absolute counts, float for proportion
    max_df= float(1.0) #int for absolute counts, float for proportion
    gamma = 0
    k = 3
    alpha = .01
    fold = 5

    input_datas = []
    alphas = [100, 10, 1, .1, .01, .001]
    for _ in alphas:
        input_datas.append(load_data(fold, phrase_len, min_df, max_df))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(X, y, xtrain, xtest, ytrain, ytest, k, gamma, alphas[i], alphas[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Model evaluations (MSE)')
    print('Data for alpha')
    print('alpha,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

    

if __name__ == '__main__':
    main()
