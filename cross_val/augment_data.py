import pandas as pd
import numpy as np
import sys

#cat cross-val_knn* > cross_vals_all.csv

def load_data(fname):
    return pd.read_csv(fname, header=0)

def main():
    use_knn = False
    if len(sys.argv) > 1 and int(sys.argv[1]) == 1:
        use_knn = True
    #From train.py
    max_dfs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    min_dfs = [0.0, 0.01, 0.1, 0.2]
    phrase_lens = [1, 3, 5, 7, 9]
    alphas = [100, 10, 1, .1, .01, .001, .0001]
    ks = [1, 5, 25, 50, 100, 150, 200, 300]
    hyper_param_last_idx = 3

    #Input data
    df = load_data('knn/cross_vals_all.csv')

    #Output dataframe
    processed_cols = []
    for i, col in enumerate(df.columns):
        if i > hyper_param_last_idx:
            processed_cols.append(col + '_mean')
            processed_cols.append(col + '_stdev')
        else:
            processed_cols.append(col)
    df_processed = pd.DataFrame(columns=processed_cols)

    #Add aggregate data to output data frame
    it = 0
    for max_df in max_dfs:
        for min_df in min_dfs:
            for phrase_len in phrase_lens:
                if not use_knn:
                    for alpha in alphas:
                        #Get subset where all hyperparameters are the same
            
                        condition = (df['max_df'] == max_df) \
                            & (df['min_df'] == min_df) \
                            & (df['phrase_len'] == phrase_len) \
                            & (df['alpha'] == alpha)
                        df_same_hyper = df[condition]

                        #Add new columns with aggreagate data to the datafram
                        new_cols = []
                        for i, col in enumerate(df_same_hyper.columns):
                            if i > hyper_param_last_idx:
                                new_cols.append(np.mean(df_same_hyper[col]))
                                new_cols.append(np.std(df_same_hyper[col], ddof=1))
                            else:
                                new_cols.append(np.median(df_same_hyper[col]))
                        df_processed.loc[it] = new_cols
                        it += 1
                else:
                    for k in ks:
                        #Get subset where all hyperparameters are the same
                
                        condition = (df['max_df'] == max_df) \
                            & (df['min_df'] == min_df) \
                            & (df['phrase_len'] == phrase_len) \
                            & (df['k'] == k)
                        df_same_hyper = df[condition]

                        #Add new columns with aggreagate data to the datafram
                        new_cols = []
                        for i, col in enumerate(df_same_hyper.columns):
                            if i > hyper_param_last_idx:
                                new_cols.append(np.mean(df_same_hyper[col]))
                                new_cols.append(np.std(df_same_hyper[col], ddof=1))
                            else:
                                new_cols.append(np.median(df_same_hyper[col]))
                        df_processed.loc[it] = new_cols
                        it += 1
    df_processed = df_processed.dropna()
    df_processed.to_csv('aggregate.csv', index=False)

if __name__ == '__main__':
    main()
