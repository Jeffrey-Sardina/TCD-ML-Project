from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import time

enlarged_axes = []
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
    surface = axes.plot_surface(x, y, z, cmap=cm.Reds, zorder=1) #saturation-based
    
    #Standardzize axes and colors
    axes.set_zlim(min_z, max_z)
    surface.set_clim(min_z, max_z)

    #Use orthographic projection
    axes.set_proj_type('ortho')

    #Label graph
    error_bars = []
    if title != None: #only true when data is enlarged
        axes.set_title(title + '\nDouble click to toggle (+) error bars')
        for i, stdev in enumerate(stdevs):
            x_loc = x_ini.iloc[i]
            y_loc = y_ini.iloc[i]
            z_loc = z_ini.iloc[i]
            error_bar = axes.plot((x_loc, x_loc), (y_loc, y_loc), (z_loc, z_loc + stdev), color='#888888', zorder=12, linewidth=2)
            error_bars.append(error_bar[0])

    axes.set_xlabel(col1_head)
    axes.set_ylabel(col2_head)
    axes.set_zlabel('MSE', rotation='horizontal')
    
    if(show_color_bar):
        color_bar = figure.colorbar(surface)
        color_bar.set_label('MSE')
        color_bar.ax.xaxis.set_label_position('top')

    return surface, min_z, max_z, error_bars

def flatten_2d(df, col1_head, col2_head, fixed1_head, fixed1, fixsed2_head, fixed2):
    col1s = df[col1_head].unique()
    col2s = df[col2_head].unique()

    df_processed = pd.DataFrame(columns=df.columns)
    it = 0
    for col1 in col1s:
        for col2 in col2s:
            #Get subset where all hyperparameters are the same
            condition = (df[col1_head] == col1) \
                & (df[col2_head] == col2) \
                & (df[fixed1_head] == fixed1) \
                & (df[fixsed2_head] == fixed2)
            restricted = df[condition]
            if len(restricted) > 0:
                df_processed.loc[it] = restricted.iloc[0]
                it += 1
    return df_processed.dropna()

def onclick(event):
    global enlarged_axes
    ax = event.inaxes
    if event.dblclick:
        for row in axes:
            for item in row:
                if ax == item:
                    new_fig = plt.figure()
                    i, j = ax.local_name
                    new_ax = new_fig.add_subplot(111, projection='3d')
                    title = fixed1_head + '=' + str(j_elems[j]) + '; ' + fixed2_head + '=' + str(i_elems[i])
                    _, _, _, error_bars = plot_3d(df, new_fig, new_ax, col1_head, col2_head, col3_head, stdev_head, fixed1_head, j_elems[j], fixed2_head, i_elems[i], title=title, show_color_bar=True)
                    new_fig.canvas.mpl_connect('button_press_event', onclick_sub)
                    new_ax.error_bars = error_bars #new attr I added (not there by default)
                    new_ax.showing_error_bars = True #new attr I added (not there by default)
                    new_ax.toggle_time = 0 #new attr I added (not there by default)
        plt.tight_layout()
        plt.show()

def onclick_sub(event):
    ax = event.inaxes
    if event.dblclick and time.time() - ax.toggle_time > 0.5:
        for error_bar in ax.error_bars:
            if ax.showing_error_bars:
                error_bar.set_alpha(0)
            else:
                error_bar.set_alpha(1)
        ax.showing_error_bars = not ax.showing_error_bars
        ax.toggle_time = time.time()
        plt.draw()

def load_data(fname):
    return pd.read_csv(fname, header=0)

def main():
    global axes, i_elems, j_elems, df, axes, col1_head, col2_head, col3_head, stdev_head, fixed1_head, fixed2_head

    df = load_data('knn/aggregate.csv')
    col1_head = 'max_df'
    col2_head = 'phrase_len'
    fixed1_head = 'k'
    fixed2_head = 'min_df'

    col3_head = 'knn_mse_mean'
    stdev_head = 'knn_mse_stdev'
    reg_type = 'kNN Regressor' 

    #From train.py
    i_elems = min_dfs = [0.0, 0.01, 0.1] #[0.0, 0.01, 0.1, 0.2]
    j_elems = ks = [1, 5, 25, 50, 100, 150, 200, 300]

    #Create figure
    #https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    surface = None
    min_z = None
    max_z = None
    figure, axes = plt.subplots(nrows=len(min_dfs), ncols=len(ks), subplot_kw={'projection':'3d'})
    for i, fixed2 in enumerate(min_dfs):
        for j, fixed1 in enumerate(ks):
            axes[i][j].local_name = (i, j) #new attr I added (not there by default)
            surface, min_z, max_z, _ = plot_3d(df, figure, axes[i][j], col1_head, col2_head, col3_head, stdev_head, fixed1_head, fixed1, fixed2_head, fixed2)

    #Add a dsingle color bar to descirbe all the figures
    #https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/

    c_left = 0.925
    c_bottom = 0.1
    c_width = 0.02
    c_height = 0.8
    colorbar_axes = figure.add_axes([c_left, c_bottom, c_width, c_height])
    colorbar = figure.colorbar(surface, cax=colorbar_axes, ticks=np.linspace(min_z, max_z, 5))
    colorbar.ax.set_yticklabels(int(x) for x in np.linspace(min_z, max_z, 5))
    colorbar.set_label('MSE')
    colorbar.ax.yaxis.set_label_position('right')

    left = 0.01
    bottom = 0.05
    width = 0.91
    height = 0.99
    plt.tight_layout(rect=(left, bottom, width, height))

    #https://stackoverflow.com/questions/57546492/multiple-plots-on-common-x-axis-in-matplotlib-with-common-y-axis-labeling
    figure.text(0.45, 0.01, 'kNN k', va='center', size=13, weight='bold')
    step = (width - c_width) / len(ks)
    for i, k in enumerate(ks):
        x_loc = i * step + step / 2
        y_loc = 0.025
        figure.text(x_loc, y_loc, str(k), va='center', weight='bold')
    
    figure.text(0.01, 0.5, 'Min Document Frequency', va='center', rotation='vertical', size=13, weight='bold')
    step = height / len(min_dfs)
    for i, min_df in enumerate(reversed(min_dfs)):
        x_loc = 0.02
        y_loc = i * step + step / 2
        figure.text(x_loc, y_loc, str(min_df), va='center', weight='bold')

    figure.text(0.4, 0.99, 'Cross-validations on ' + reg_type, va='center', size=15, weight='bold')
    figure.text(0.45, 0.97, 'Double click to enlarge', va='center', size=13, weight='bold')

    #https://matplotlib.org/3.2.1/users/event_handling.html
    figure.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

if __name__ == '__main__':
    main()
