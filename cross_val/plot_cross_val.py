from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from matplotlib import cm

axes = None
i_elems = None
j_elems = None
df = None
col1_head = None
col2_head = None
col3_head = None
stdev_head = None
fixed1_head = None
fixed2_head = None
def plot_3d(df, figure, axes, col1_head, col2_head, col3_head, stdev_head, fixed1_head, fixed1, fixed2_head, fixed2, title=None, show_color_bar=False):
    #Z range (must be run on pre-preprocessed data to ensure it is the same for all graphs)
    min_z = min(df[col3_head])
    max_z = max(df[col3_head])

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

    #Plot data
    #https://stackoverflow.com/questions/57123749/plotting-a-3d-line-intersecting-a-surface-in-mplot3d-matplotlib-python
    surface = axes.plot_surface(x, y, z, edgecolor='#ffffff', cmap=cm.inferno, zorder=1) #perceptually uniform
    
    #Standardzize axes and colors
    axes.set_zlim(min_z, max_z)
    surface.set_clim(min_z, max_z)

    #Use orthographic projection
    axes.set_proj_type('ortho')

    #Label graph
    if title != None: #only true when data is enlarged
        axes.set_title(title)
        for i, stdev in enumerate(stdevs):
            x_loc = x_ini.iloc[i]
            y_loc = y_ini.iloc[i]
            z_loc = z_ini.iloc[i]
            axes.plot((x_loc, x_loc), (y_loc, y_loc), (z_loc, z_loc + stdev), color='#aaaaaa', zorder=12, linewidth=2)
    axes.set_xlabel(col1_head)
    axes.set_ylabel(col2_head)
    axes.set_zlabel('MSE')

    
    if(show_color_bar):
        figure.colorbar(surface)

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

def onclick(event):
    ax = event.inaxes
    if event.dblclick:
        for row in axes:
            for item in row:
                if ax == item:
                    new_fig = plt.figure()
                    i, j = ax.local_name
                    new_ax = new_fig.add_subplot(111, projection='3d')
                    title = fixed1_head + '=' + str(j_elems[j]) + '; ' + fixed2_head + '=' + str(i_elems[i])
                    plot_3d(df, new_fig, new_ax, col1_head, col2_head, col3_head, stdev_head, fixed1_head, j_elems[j], fixed2_head, i_elems[i], title=title, show_color_bar=True)
        plt.tight_layout()
        plt.show()

def load_data(fname):
    return pd.read_csv(fname, header=0)

def main():
    global axes, i_elems, j_elems, df, axes, col1_head, col2_head, col3_head, stdev_head, fixed1_head, fixed2_head

    df = load_data('aggregate.csv')
    col1_head = 'max_df'
    col2_head = 'phrase_len'
    fixed1_head = 'alpha'
    fixed2_head = 'min_df'

    col3_head = 'lin_reg_mse_mean'
    stdev_head = 'lin_reg_mse_stdev'
    reg_type = 'Linear Regression'

    '''col3_head = 'lasso_mse_mean'
    stdev_head = 'lasso_mse_stdev'
    reg_type = 'Lasso Regression'

    col3_head = 'ridge_mse_mean'
    stdev_head = 'ridge_mse_stdev'
    reg_type = 'Ridge Regression' '''

    #From train.py
    i_elems = min_dfs = [0.0, 0.01, 0.1] #[0.0, 0.01, 0.1, 0.2]
    j_elems = alphas = [100, 10, 1, .1, .01, .001, .0001]

    #https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    #Create figure
    figure, axes = plt.subplots(nrows=len(min_dfs), ncols=len(alphas), subplot_kw={'projection':'3d'})
    for i, fixed2 in enumerate(min_dfs):
        for j, fixed1 in enumerate(alphas):
            axes[i][j].local_name = (i, j)
            show_color_bar = False
            plot_3d(df, figure, axes[i][j], col1_head, col2_head, col3_head, stdev_head, fixed1_head, fixed1, fixed2_head, fixed2, show_color_bar=show_color_bar)

    #https://stackoverflow.com/questions/57546492/multiple-plots-on-common-x-axis-in-matplotlib-with-common-y-axis-labeling
    figure.text(0.45, 0.01, 'Regularization Alpha', va='center', size=13)
    step = 1 / len(alphas)
    for i, alpha in enumerate(alphas):
        x_loc = i * step + step / 2
        y_loc = 0.025
        figure.text(x_loc, y_loc, str(alpha), va='center')
    
    figure.text(0.01, 0.5, 'Min Document Frequency', va='center', rotation='vertical', size=13)
    step = 1 / len(min_dfs)
    for i, min_df in enumerate(reversed(min_dfs)):
        x_loc = 0.02
        y_loc = i * step + step / 2
        figure.text(x_loc, y_loc, str(min_df), va='center')

    figure.text(0.4, 0.99, 'Cross-validations on ' + reg_type, va='center', size=15)
    figure.text(0.45, 0.97, 'Double click to enlarge', va='center', size=13)

    #https://matplotlib.org/3.2.1/users/event_handling.html
    figure.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout(rect=(0.01, 0.05, 0.99, 0.99))
    plt.show()

if __name__ == '__main__':
    main()
