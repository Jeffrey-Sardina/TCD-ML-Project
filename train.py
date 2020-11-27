from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import random
import sample

###########################################################
# Glocal variables
###########################################################

#Sampling and evaluating params
fold = 5
total_len = 2000
num_samples = 20

#Data variables
documents = None
y = None

###########################################################
# General helper code
###########################################################

def create_input_array(documents, phrase_len, min_df, max_df):
    '''
    This function creates a bag of words (X / input) matrix to be used in the model based on the given documents
    '''
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, phrase_len), min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(documents)
    return X

def process_data(documents, y, fold, phrase_len, min_df, max_df):
    '''
    This function splits data between training and test sets, which are used in running training and evaluation.
    '''
    X = create_input_array(documents, phrase_len, min_df, max_df)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=1 / fold)
    return X, xtrain, xtest, ytrain, ytest

def evaluations(xtrain, xtest, ytrain, ytest, alpha):
    '''
    This code runs evaluations of various linear regression models against a baseline. The alpha parameter is the multiplier to the regularization penalty in Lasso and Ridge Regression.
    '''
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

    return lin_reg_mse, lasso_mse, ridge_mse, baseline_mse

###########################################################
# Post-cross-validation model evaludation code
###########################################################

def eval_model(phrase_len, min_df, max_df, alpha, model_name):
    '''
    This function trains and evaluates a model on the given data. It also compares it to a baseline
    '''
    baseline = DummyRegressor(strategy="mean").fit(xtrain, ytrain)
    documents, y = sample.run_sample(total_len, num_samples)
    _, xtrain, xtest, ytrain, ytest = process_data(documents, y, fold, phrase_len, min_df, max_df)
    if model_name == 'LinearRegression':
        model = LinearRegression().fit(xtrain, ytrain)
    elif model_name == 'Lasso':
        model = Lasso(alpha=alpha).fit(xtrain, ytrain)
    else:
        model = Ridge(alpha=alpha).fit(xtrain, ytrain)

    model_mse = mean_squared_error(ytest, model.predict(xtest))
    baseline_mse = mean_squared_error(ytest, baseline.predict(xtest))

    print()
    print('model_mse,baseline_mse')
    print(model_mse, baseline_mse, sep=',')

###########################################################
# Cross-validation code
###########################################################

def cross_validations(out_file_name):
    '''
    This function runs corss-validations for all valid combinations of a range of hyperparameters. Code is multiprocessed to speed it up, and results are written to a CSV file.
    '''
    global documents, y
    documents, y = sample.run_sample(total_len, num_samples)

    max_dfs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    min_dfs = [0.0, 0.01, 0.1, 0.2]
    phrase_lens = [1, 3, 5, 7, 9]

    args_lists = []
    for max_df in max_dfs:
        for min_df in min_dfs:
            for phrase_len in phrase_lens:
                args_lists.append([max_df, min_df, phrase_len])
    random.shuffle(args_lists) #Balance out work load--larger phrase len => much longer to train

    processes = 8
    arg_divs = [args_lists[i * len(args_lists) // processes : (i + 1) * len(args_lists) // processes] for i in range(processes)]
    pool = Pool(processes=processes, initializer=init_args, initargs=[documents, y])
    result = pool.map(eval_all, arg_divs)
    pool.close()

    with open(out_file_name, 'w') as out:
        print('max_df,min_df,phrase_len,alpha,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse', file=out)
        for data_list in result:
            for line in data_list:
                print(line, file=out)

def eval_all(text_params):
    '''
    This function does the 'brunt work' of cross-validation--given a list of max_df, min_df, and phrase_len tuples (text_params) as hyperparameters and using a range of alpha (regularization multiplier) values, it runs evaluations on all of them and returns the MSEs of the results.
    '''
    alphas = [100, 10, 1, .1, .01, .001, .0001]
    data_strs = []
    i = 0
    for max_df, min_df, phrase_len in text_params:
        try: #Somer param combos are not valid and wilol throw an error
            _, xtrain, xtest, ytrain, ytest = process_data(documents, y, fold, phrase_len, min_df, max_df)
            for alpha in alphas:
                print(i)
                i += 1
                
                data = evaluations(xtrain, xtest, ytrain, ytest, alpha)
                data_str = str(max_df) + ',' + str(min_df) + ',' + str(phrase_len) + ',' + str(alpha) + ','
                data_str += ','.join(str(x) for x in data)
                data_strs.append(data_str)
        except:
                pass
                print('error')
    return data_strs

def init_args(local_documents, local_y):
    '''
    This function is used to provide global parameters to each process when multiprocessing
    '''
    global documents, y
    documents = local_documents
    y = local_y

###########################################################
# Startup code
###########################################################

def main():
    #Run cross-validation several times to get enough data to calc standard deviations
    n = 10
    for i in range(n):
        cross_validations('cross-val_' + str(i) + '.csv')

    #Construct a model based on the optimal hyperparameters and model type
    max_df = 0
    min_df = 0
    phrase_len = 0
    alpha = 0


if __name__ == '__main__':
    main()
