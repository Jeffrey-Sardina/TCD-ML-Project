from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import random
import sample

#Global params
fold = 5
total_len = 2000
num_samples = 20

#Data
documents = None
y = None

def create_input_array(documents, phrase_len, min_df, max_df):
    '''
    This function creates a bag of words (X / input) matrix to be used in the model based on the given documents
    '''
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, phrase_len), min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(documents)
    return X

def process_data(documents, y, fold, phrase_len, min_df, max_df):
    #Gather and process data into a form that we can run ML on
    X = create_input_array(documents, phrase_len, min_df, max_df)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=1 / fold)
    return X, xtrain, xtest, ytrain, ytest

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

    return lin_reg_mse, lasso_mse, ridge_mse, baseline_mse

def init_args(local_documents, local_y):
    global documents, y
    documents = local_documents
    y = local_y

def main():
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

    with open('cross-val.csv', 'w') as out:
        print('max_df,min_df,phrase_len,alpha,lin_reg_mse,lasso_mse,ridge_mse,baseline_mse', file=out)
        for data_list in result:
            for line in data_list:
                print(line, file=out)

def eval_model(phrase_len, min_df, max_df, alpha):
    documents, y = sample.run_sample(total_len, num_samples)
    _, xtrain, xtest, ytrain, ytest = process_data(documents, y, fold, phrase_len, min_df, max_df)
    data = evaluations(xtrain, xtest, ytrain, ytest, alpha)
    data_str = ','.join(str(x) for x in data) + '\n'

    print()
    print('lin_reg_mse,lasso_mse,ridge_mse,baseline_mse')
    print(data_str)

def eval_all(text_params):
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
                data_str += ','.join(str(x) for x in data) + '\n'
                data_strs.append(data_str)
        except:
                pass
                print('error')
    return data_strs

if __name__ == '__main__':
    main()
