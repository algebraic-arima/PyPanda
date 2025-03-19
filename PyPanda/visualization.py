import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def PMTHeatmapper(freq_df, xtick, ytick, param):
    plt.figure(figsize=(15, 12))
    freq_df = np.flip(freq_df, axis=0)
    sns.heatmap(freq_df,
                cmap='viridis',
                fmt=".0f",
                linewidths=.5,
                square=True,
                xticklabels=xtick,
                yticklabels=ytick)
    plt.title(f'PMT Heatmap {param}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
    plt.close()


def PMTImshow(freq_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(freq_df, cmap='viridis', fmt=".0f", linewidths=.5, square=True)
    plt.title(f'PMT Heatmap ')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # plt.savefig(image_path)
    plt.show()
    plt.close()


def PMT3DScatter(matrix, color):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    z = matrix
    ax.scatter(x, y, z, color=color, s=10)
