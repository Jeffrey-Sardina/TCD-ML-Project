import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import sample

#Global params
fold = 5
total_len = 2000
num_samples = 20

def create_input_array(documents, phrase_len, min_df, max_df):
    '''
    This function creates a bag of words (X / input) matrix to be used in the model based on the given documents
    '''
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, phrase_len), min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(documents)
    return X

def load_data(fold, phrase_len, min_df, max_df):
    #Gather and process data into a form that we can run ML on
    documents, y = sample.run_sample(total_len, num_samples)
    X = create_input_array(documents, phrase_len, min_df, max_df)

    #Split into train and test segments
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=1 / fold)
    return X, y, xtrain, xtest, ytrain, ytest

def evaluations(xtrain, xtest, ytrain, ytest, alpha,):
    #Create models
    lin_reg_model = LinearRegression().fit(xtrain, ytrain)
    lasso_model = Lasso(alpha=alpha).fit(xtrain, ytrain)
    ridge_model = Ridge(alpha=alpha).fit(xtrain, ytrain)
    baseline = DummyRegressor(strategy="mean").fit(xtrain, ytrain)

    #Evaluate models
    lin_reg_mse = mean_squared_error(ytest, lin_reg_model.predict(xtest))
    lasso_mse = mean_squared_error(ytest, lasso_model.predict(xtest))
    ridge_mse = mean_squared_error(ytest, ridge_model.predict(xtest))
    baseline_mse = mean_squared_error(ytest, baseline.predict(xtest))

    return lin_reg_mse,lasso_mse,ridge_mse,baseline_mse

def main():
    phrase_len = 1
    min_df = int(1) #int for absolute counts, float for proportion
    max_df= float(1.0) #int for absolute counts, float for proportion
    alpha = .01

    eval_all()

    # eval_model(phrase_len, min_df, max_df, alpha)
    
    '''max_dfs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    eval_max_df(phrase_len, min_df, alpha, max_dfs)
    print()

    min_dfs = [0.0, 0.01, 0.1, 0.2]
    eval_min_df(phrase_len, max_df, alpha, min_dfs)
    print()

    phrase_lens = [1, 3, 5, 7, 9]
    eval_phrase_len(min_df, max_df, alpha, phrase_lens)
    print()
    
    alphas = [100, 10, 1, .1, .01, .001]
    eval_alpha(phrase_len, min_df, max_df, alphas)
    print()'''

def eval_model(phrase_len, min_df, max_df, alpha):
    X, y, xtrain, xtest, ytrain, ytest = load_data(fold, phrase_len, min_df, max_df)
    data = evaluations(xtrain, xtest, ytrain, ytest, alpha, None)
    data_str = ','.join(str(x) for x in data) + '\n'

    print()
    print('ignore,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

def eval_all():
    max_dfs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    min_dfs = [0.0, 0.01, 0.1, 0.2]
    phrase_lens = [1, 3, 5, 7, 9]
    alphas = [100, 10, 1, .1, .01, .001]

    input_datas = []
    data_str = ''
    i = 0
    for max_df in max_dfs:
        for min_df in min_dfs:
            for phrase_len in phrase_lens:
                X, y, xtrain, xtest, ytrain, ytest = load_data(fold, phrase_len, min_df, max_df)
                for alpha in alphas:
                    print(i)
                    i += 1
                    data = evaluations(xtrain, xtest, ytrain, ytest, alpha)
                    data_str += ','.join([str(max_df), str(min_df), str(phrase_len), str(alpha)]) + ','
                    data_str += ','.join(str(x) for x in data) + '\n'
    print('max_df,min_df,phrase_len,alpha,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

'''def eval_max_df(phrase_len, min_df, alpha, max_dfs):
    input_datas = []
    for val in max_dfs:
        input_datas.append(load_data(fold, phrase_len, min_df, val))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(xtrain, xtest, ytrain, ytest, alpha, max_dfs[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Data for max DF')
    print('max_df,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

def eval_min_df(phrase_len, max_df, alpha, min_dfs):
    input_datas = []
    for val in min_dfs:
        input_datas.append(load_data(fold, phrase_len, val, max_df))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(xtrain, xtest, ytrain, ytest, alpha, min_dfs[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Data for min DF')
    print('min_df,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

def eval_phrase_len(min_df, max_df, alpha, phrase_lens):
    input_datas = []
    for val in phrase_lens:
        input_datas.append(load_data(fold, val, min_df, max_df))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(xtrain, xtest, ytrain, ytest, alpha, phrase_lens[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Data for phrase length')
    print('phrase_len,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

def eval_alpha(phrase_len, min_df, max_df, alphas):
    input_datas = []
    for _ in alphas:
        input_datas.append(load_data(fold, phrase_len, min_df, max_df))

    data_str = ''
    for i, (X, y, xtrain, xtest, ytrain, ytest) in enumerate(input_datas):
        data = evaluations(xtrain, xtest, ytrain, ytest, alphas[i], alphas[i])
        data_str += ','.join(str(x) for x in data) + '\n'
    print('Data for alpha')
    print('alpha,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)'''

if __name__ == '__main__':
    main()
