import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

import PyPanda as pp

data = np.load(r'../../../DATA/Geant4/data_15wtest.npy')
load_encoder = pp.LoadEncoder(model_path=r'../../../Model/Encoder/model/encoder2d.pth')
load_pmt2pos = pp.LoadPMT2POS(model_path=r'../../denoise_CNN/model/conv_model_combined_transfer.pth')

G4_pos_list = []
clean_pos_list = []
print(np.size(data, 0))
for index in range(np.size(data, 0)):
    print(index)

    data1 = data[index, :56 ** 2].reshape(56, 56)
    data2 = data[index, 56 ** 2:56 ** 2 * 2].reshape(56, 56)
    data1 = data1 / data1.max()
    data2 = data2 / data2.max()

    denoise_data1 = load_encoder.reconstruct(data1).numpy().reshape(56, 56)
    denoise_data2 = load_encoder.reconstruct(data2).numpy().reshape(56, 56)

    denoise_data1 = denoise_data1 / denoise_data1.max()
    denoise_data2 = denoise_data2 / denoise_data2.max()

    G4_pos = load_pmt2pos.reconstruct(data1, data2).numpy()
    clean_pos = load_pmt2pos.reconstruct(denoise_data1, denoise_data2).numpy()

    G4_pos_list.append(G4_pos)
    clean_pos_list.append(clean_pos)

# 将列表转换为numpy数组
G4_pos_array = np.array(G4_pos_list, dtype=float)
clean_pos_array = np.array(clean_pos_list, dtype=float)

# 保存数据到文件
np.save('G4_pos_15w.npy', G4_pos_array)
np.save('G4clean_15w.npy', clean_pos_array)


def calculate_error(data_pre, data):
    error = data_pre - data
    distances = np.sqrt(np.sum((data_pre - data) ** 2, axis=1))
    mean_distance = np.mean(distances)
    return error, mean_distance


G4_pos_array = G4_pos_array.reshape(np.size(G4_pos_array, 0), 2)
clean_pos_array = clean_pos_array.reshape(np.size(G4_pos_array, 0), 2)

# 计算误差
true_pos = data[:, -3:-1] / 100
classic_pos_error, classic_pos_mean = calculate_error(G4_pos_array, true_pos)
clean_pos_error, clean_pos_mean = calculate_error(clean_pos_array, true_pos)

# 绘图
classic_pos_density = gaussian_kde(classic_pos_error.T)
clean_pos_density = gaussian_kde(clean_pos_error.T)

# 设置图形和子图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

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
mean_error = [classic_pos_mean, clean_pos_mean]
# 绘制每个子图的密度轮廓图
for i, (ax, pos_error, x_range, y_range) in enumerate(
    zip(axs, [classic_pos_error, clean_pos_error], x_ranges, y_ranges)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    pos_density = gaussian_kde(pos_error.T)
    Z = pos_density(np.vstack([X.ravel(), Y.ravel()]))
    ax.contourf(X, Y, Z.reshape(X.shape), cmap='viridis', levels=20)
    # ax.scatter(pos_error[:, 0], pos_error[:, 1], color='r', s=1)  # 原始点
    ax.set_title(titles[i] + f"mean:{mean_error[i] * 100}mm")

plt.tight_layout()

plt.savefig("G4_new15w.png")
