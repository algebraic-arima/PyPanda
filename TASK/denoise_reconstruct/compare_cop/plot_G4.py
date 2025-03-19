import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def calculate_error(data_pre, data):
    error = data_pre - data
    distances = np.sqrt(np.sum((data_pre - data) ** 2, axis=1))
    mean_distance = np.mean(distances)
    return error,mean_distance


data = np.load(r'../../../DATA/Geant4/data_15wtest.npy')
classic_pos_array=np.load('G4_pos_15w.npy')
clean_pos_array=np.load('G4clean_15w.npy')

classic_pos_array=classic_pos_array.reshape(np.size(classic_pos_array,0),3)
clean_pos_array=clean_pos_array.reshape(np.size(classic_pos_array,0),3)

classic_pos_array=classic_pos_array[:,0:2]
clean_pos_array=clean_pos_array[:,0:2]

# 计算误差
true_pos = data[:, -3:-1] / 100
classic_pos_error ,classic_pos_mean= calculate_error(classic_pos_array, true_pos)
clean_pos_error ,clean_pos_mean= calculate_error(clean_pos_array, true_pos)


# 绘图
classic_pos_density = gaussian_kde(classic_pos_error.T)
clean_pos_density = gaussian_kde(clean_pos_error.T)


# 设置图形和子图
fig, axs = plt.subplots(1, 2, figsize=(12,6))

# 为每个子图设置不同的范围
x_ranges = [
    (classic_pos_error[:, 0].min(), classic_pos_error[:, 0].max()),
    (clean_pos_error[:, 0].min(), clean_pos_error[:, 0].max()),
]
y_ranges = [
    (classic_pos_error[:, 1].min(), classic_pos_error[:, 1].max()),
    (clean_pos_error[:, 1].min(), clean_pos_error[:, 1].max()),
]
titles = ['G4_pos_error', 'G4clean_pos_error']
mean_error=[classic_pos_mean,clean_pos_mean]
# 绘制每个子图的密度轮廓图
for i, (ax, pos_error, x_range, y_range) in enumerate(zip(axs, [classic_pos_error, clean_pos_error], x_ranges, y_ranges)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    pos_density = gaussian_kde(pos_error.T)
    Z = pos_density(np.vstack([X.ravel(), Y.ravel()]))
    ax.contourf(X, Y, Z.reshape(X.shape), cmap='viridis', levels=20)
    # ax.scatter(pos_error[:, 0], pos_error[:, 1], color='r', s=1)  # 原始点
    ax.set_title(titles[i]+f"mean:{mean_error[i]*100}mm")

plt.tight_layout()

plt.savefig("G4_new15w.png")
