from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from matplotlib import cm

def plot_3d(df, axes, col1_head, col2_head, col3_head, stdev_head, fixed1_head, fixed1, fixed2_head, fixed2):
    #Add data
    flat = flatten_2d(df, col1_head, col2_head, fixed1_head, fixed1, fixed2_head, fixed2)
    x_ini = flat[col1_head]
    y_ini = flat[col2_head]
    z_ini = flat[col3_head]
    stdevs = flat[stdev_head]

    col1_len = len(flat[col1_head].unique())
    col2_len = len(flat[col2_head].unique())

    x = x_ini.to_numpy().reshape(col1_len, col2_len)
    y = y_ini.to_numpy().reshape(col1_len, col2_len)
    z = z_ini.to_numpy().reshape(col1_len, col2_len)
    axes.plot_surface(x, y, z, cmap='inferno') #color='#ff0000aa')

    for i, stdev in enumerate(stdevs):
        x_loc = x_ini.iloc[i]
        y_loc = y_ini.iloc[i]
        z_loc = z_ini.iloc[i]
        axes.plot((x_loc, x_loc), (y_loc, y_loc), (z_loc, z_loc + stdev), color='#000000')

    #Label graph
    axes.set_xlabel(col1_head)
    axes.set_ylabel(col2_head)
    axes.set_zlabel(col3_head)
    axes.set_title('Initial data')

def flatten_2d(df, col1_head, col2_head, fixed1_head, fixed1, fixsed2_head, fixed2):
    col1s = df[col1_head].unique()
    col2s = df[col2_head].unique()
    hyper_param_last_idx = 3

    df_processed = pd.DataFrame(columns=df.columns)
    it = 0
    for col1 in col1s:
        for col2 in col2s:
            #Get subset where all hyperparameters are the same
            condition = (df[col1_head] == col1) \
                & (df[col2_head] == col2) \
                & (df[fixed1_head] == fixed1) \
                & (df[fixsed2_head] == fixed2)            
            df_processed.loc[it] = df[condition].iloc[0]
            it += 1
    return df_processed.dropna()

def flatten_2d_avg(df, col1_head, col2_head):
    col1s = df[col1_head].unique()
    col2s = df[col2_head].unique()
    hyper_param_last_idx = 3

    df_processed = pd.DataFrame(columns=df.columns)
    it = 0
    for col1 in col1s:
        for col2 in col2s:
            #Get subset where all hyperparameters are the same
            condition = (df[col1_head] == col1) \
                & (df[col2_head] == col2)
            df_same_hyper = df[condition]

            #Add new columns with aggreagate data to the datafram
            new_cols = []
            for i, col in enumerate(df_same_hyper.columns):
                if i > hyper_param_last_idx:
                    new_cols.append(np.mean(df_same_hyper[col]))
                else:
                    new_cols.append(np.median(df_same_hyper[col]))
            df_processed.loc[it] = new_cols
            it += 1
    return df_processed.dropna()

def load_data(fname):
    return pd.read_csv(fname, header=0)

def main():
    df = load_data('aggregate.csv')
    col1_head = 'max_df'
    col2_head = 'phrase_len'
    col3_head = 'lin_reg_mse_mean'
    stdev_head = 'lin_reg_mse_stdev'
    fixed1_head = 'alpha'
    fixed2_head = 'min_df'

    #From train.py
    min_dfs = [0.0, 0.01, 0.1] #[0.0, 0.01, 0.1, 0.2]
    alphas = [100, 10, 1, .1, .01, .001, .0001]

    #https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    #Create figure
    figure, axes = plt.subplots(nrows=len(min_dfs), ncols=len(alphas), subplot_kw={'projection':'3d'})
    for i, fixed2 in enumerate(min_dfs):
        for j, fixed1 in enumerate(alphas):
            try:
                plot_3d(df, axes[i][j], col1_head, col2_head, col3_head, stdev_head, fixed1_head, fixed1, fixed2_head, fixed2)
            except:
                pass
    plt.show()

if __name__ == '__main__':
    main()
