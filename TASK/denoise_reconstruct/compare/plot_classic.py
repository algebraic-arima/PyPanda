import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def calculate_error(data_pre, data):
    error = data_pre - data
    distances = np.sqrt(np.sum((data_pre - data) ** 2, axis=1))
    mean_distance = np.mean(distances)
    return error,mean_distance


data = np.load(r'../../../DATA/Classic/data_classic_8e3.npy')
classic_pos_array=np.load('classic_pos_rescale.npy')
clean_pos_array=np.load('clean_pos_rescale.npy')
noise_pos_array=np.load('noise_pos_rescale.npy')
classic_pos_array=classic_pos_array.reshape(9261,3)
clean_pos_array=clean_pos_array.reshape(9261,3)
noise_pos_array=noise_pos_array.reshape(9261,3)
classic_pos_array=classic_pos_array[:,0:2]
clean_pos_array=clean_pos_array[:,0:2]
noise_pos_array=noise_pos_array[:,0:2]
# 计算误差
true_pos = data[:, -3:-1] / 100
classic_pos_error ,classic_pos_mean= calculate_error(classic_pos_array, true_pos)
clean_pos_error ,clean_pos_mean= calculate_error(clean_pos_array, true_pos)
noise_pos_error ,noise_pos_mean= calculate_error(noise_pos_array, true_pos)



# 绘图
classic_pos_density = gaussian_kde(classic_pos_error.T)
clean_pos_density = gaussian_kde(clean_pos_error.T)
noise_pos_density = gaussian_kde(noise_pos_error.T)

# 设置图形和子图
fig, axs = plt.subplots(1, 3, figsize=(18,6))

# 为每个子图设置不同的范围
x_ranges = [
    (classic_pos_error[:, 0].min(), classic_pos_error[:, 0].max()),
    (clean_pos_error[:, 0].min(), clean_pos_error[:, 0].max()),
    (noise_pos_error[:, 0].min(), noise_pos_error[:, 0].max())
]
y_ranges = [
    (classic_pos_error[:, 1].min(), classic_pos_error[:, 1].max()),
    (clean_pos_error[:, 1].min(), clean_pos_error[:, 1].max()),
    (noise_pos_error[:, 1].min(), noise_pos_error[:, 1].max())
]
titles = ['classic_pos_error', 'clean_pos_error', 'noise_pos_error']
mean_error=[classic_pos_mean,clean_pos_mean,noise_pos_mean]
# 绘制每个子图的密度轮廓图
for i, (ax, pos_error, x_range, y_range) in enumerate(zip(axs, [classic_pos_error, clean_pos_error, noise_pos_error], x_ranges, y_ranges)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    pos_density = gaussian_kde(pos_error.T)
    Z = pos_density(np.vstack([X.ravel(), Y.ravel()]))
    ax.contourf(X, Y, Z.reshape(X.shape), cmap='viridis', levels=20)
    # ax.scatter(pos_error[:, 0], pos_error[:, 1], color='r', s=1)  # 原始点
    ax.set_title(titles[i]+f"mean:{mean_error[i]*100}mm")

plt.tight_layout()
plt.savefig("Classic_rescale.png")


