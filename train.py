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

def lasso_cross_validate_alpha(alphas, fold, X, y):
    kf = KFold(n_splits=fold, shuffle=True)
    mean_errors = []
    std_errors = []
    for alpha in alphas:
        model = Lasso(alpha=alpha)
        MSEs = []
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            MSEs.append(mean_squared_error(y[test], ypred))
        MSEs = np.array(MSEs)
        mean_errors.append(MSEs.mean())
        std_errors.append(MSEs.std())
    return mean_errors, std_errors

def ridge_cross_validate_alpha(alphas, fold, X, y):
    kf = KFold(n_splits=fold, shuffle=True)
    mean_errors = []
    std_errors = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        MSEs = []
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            MSEs.append(mean_squared_error(y[test], ypred))
        MSEs = np.array(MSEs)
        mean_errors.append(MSEs.mean())
        std_errors.append(MSEs.std())
    return mean_errors, std_errors

def kNN_cross_validate_k(ks, gamma, fold, X, y):
    kf = KFold(n_splits=fold, shuffle=True)
    mean_errors = []
    std_errors = []
    for k in ks:
        model = KNeighborsRegressor(n_neighbors=k, weights=create_gaussian_kernel_function(gamma))
        MSEs = []
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            MSEs.append(mean_squared_error(y[test], ypred))
        MSEs = np.array(MSEs)
        mean_errors.append(MSEs.mean())
        std_errors.append(MSEs.std())
    return mean_errors, std_errors

def kNN_cross_validate_gamma(gammas, k, fold, X, y):
    kf = KFold(n_splits=fold, shuffle=True)
    mean_errors = []
    std_errors = []
    for gamma in gammas:
        model = KNeighborsRegressor(n_neighbors=k, weights=create_gaussian_kernel_function(gamma))
        MSEs = []
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            MSEs.append(mean_squared_error(y[test], ypred))
        MSEs = np.array(MSEs)
        mean_errors.append(MSEs.mean())
        std_errors.append(MSEs.std())
    return mean_errors, std_errors

def plot_cross_validation(params, param_name, mean_errors, std_errors, title):
    plt.cla()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.errorbar(params, mean_errors, yerr=std_errors, linewidth=1)
    plt.xlabel(param_name)
    plt.ylabel('Mean square error')
    plt.title(title)
    plt.savefig(title + '.png', bbox_inches='tight')

def load_data(fold, phrase_len, min_df, max_df):
    #Gather and process data into a form that we can run ML on
    documents, y = sample.main()
    X = create_input_array(documents, phrase_len, min_df, max_df)

    #Split into train and test segments
    indices = np.arange(X.size)
    train, test = train_test_split(indices, test_size=1 / fold)
    return X, y, train, test

def cross_validations(fold, X, y):
    alphas = []
    mean_errors, std_errors = lasso_cross_validate_alpha(alphas, fold, X, y)
    plot_cross_validation(alphas, 'alpha', mean_errors, std_errors, 'Cross-validation of Lasso alpha')

    alphas = []
    mean_errors, std_errors = ridge_cross_validate_alpha(alphas, fold, X, y)
    plot_cross_validation(alphas, 'alpha', mean_errors, std_errors, 'Cross-validation of Ridge alpha')

    ks = []
    gamma = 0
    mean_errors, std_errors = kNN_cross_validate_k(ks, gamma, fold, X, y)
    plot_cross_validation(ks, 'k', mean_errors, std_errors, 'Cross-validation of kNN with gamma=' + str(gamma))

    gammas = []
    k = 0
    mean_errors, std_errors = kNN_cross_validate_k(ks, gamma, fold, X, y)
    plot_cross_validation(gammas, 'gamma', mean_errors, std_errors, 'Cross-validation of kNN with k=' + str(gamma))

def evaluations(X, y, train, test, k, gamma, alpha):
    #Create models
    knn_model = KNeighborsRegressor(n_neighbors=k, weights=create_gaussian_kernel_function(gamma)).fit(X[train], y[train])
    lin_reg_model = LinearRegression().fit(X[train], y[train])
    lasso_model = Lasso(alpha=alpha).fit(X[train], y[train])
    ridge_model = Ridge(alpha=alpha).fit(X[train], y[train])
    baseline = DummyRegressor(strategy="mean").fit(X[train], y[train])

    #Evaluate models
    knn_mse = mean_squared_error(y[test], knn_model.predict(X[test]))
    lin_reg_mse = mean_squared_error(y[test], lin_reg_model.predict(X[test]))
    lasso_mse = mean_squared_error(y[test], lasso_model.predict(X[test]))
    ridge_mse = mean_squared_error(y[test], ridge_model.predict(X[test]))
    baseline_mse = mean_squared_error(y[test], baseline.predict(X[test]))

    print('Model evaluations (MSE)')
    print('knn_mse,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(knn_mse,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse, sep=',')
    print()

def main():
    #hyperparameters
    #to tf-idf tokenizer
    phrase_len = 5
    min_df = int(1) #int for absolute counts, float for proportion
    max_df= float(1.0) #int for absolute counts, float for proportion

    #To kNN
    gamma = 0
    k = 3

    #To lasso and ridge regession
    alpha = 1

    #Parameters
    #to kFold cross-validation
    fold = 5

    X, y, train, test = load_data(fold, phrase_len, min_df, max_df)
    print(X)

    '''#generate various input matricies based on the hyperparameters to cross-validate them
    input_datas = []
    phrase_lens = []
    for val in range(phrase_len):
        input_datas.append(load_data(fold, val, min_df, max_df))

    min_dfs = []
    for val in range(min_dfs):
        input_datas.append(load_data(fold, phrase_len, val, max_df))

    max_dfs = []
    for val in range(max_dfs):
        input_datas.append(load_data(fold, phrase_len, min_df, val))

    #Validate and evaluate models
    for X, y, train, test in input_datas:
        cross_validations(fold, X, y)
        evaluations(X, y, train, test, k, gamma, alpha)'''

if __name__ == '__main__':
    main()
