from matplotlib import pyplot as plt
import numpy as np

import PyPanda as pp


def add_noise_to_data(data, data_z, num_noise_realizations, mode):
    noisy_data_list = []
    for i in range(np.size(data, 0)):
        z = data_z[i]
        if mode == 'top':
            num_photon = (3.01389248e-02 * z ** 2 + 6.61948458e+01 * z + 4.61519056e+04) / 100 * 15
        else:
            num_photon = (8.08150300e-02 * z ** 2 - 6.62960040e+01 * z + 3.49185111e+04) / 100 * 15
        for _ in range(num_noise_realizations):
            noise = np.random.poisson(data[i, :, :] / np.sum(data[i, :, :]) * num_photon)
            noisy_data_list.append(noise / num_photon * np.sum(data[i, :, :]))
    return np.array(noisy_data_list)


index = 22
# data = np.load(r'../../DATA/Classic/data_classic_8e3.npy')
data = np.load(r'../../DATA/Geant4/data_100w.npy')
dataz = np.zeros(1)

dataz[0] = data[index, -1]
data1 = data[index, :56 ** 2].reshape(1, 56, 56)
data2 = data[index, 56 ** 2:56 ** 2 * 2].reshape(1, 56, 56)
data1 = data1 / data1.max()
data2 = data2 / data2.max()
# data_noise1=data1*(1+np.random.normal(0, 0.1, data1.shape))
# data_noise2=data2*(1+np.random.normal(0, 0.1, data2.shape))
data_noise1 = add_noise_to_data(data1, dataz, 1, 'top').reshape(56, 56)
data_noise2 = add_noise_to_data(data2, dataz, 1, 'bot').reshape(56, 56)
# data_noise1 = data_noise1 *np.sum(data1)/np.sum(data_noise1)
# data_noise2 = data_noise2 *np.sum(data2)/np.sum(data_noise2)

# data_noise1 = data_noise1 / data_noise1.max()
# data_noise2 = data_noise2 / data_noise2.max()


load_encoder = pp.LoadEncoder()
denoise_data1 = load_encoder.reconstruct(data_noise1).numpy().reshape(56, 56)
denoise_data2 = load_encoder.reconstruct(data_noise2).numpy().reshape(56, 56)
# denoise_data1 = load_encoder.reconstruct(data_noise1).numpy().reshape(56, 56)
# denoise_data2 = load_encoder.reconstruct(data_noise2).numpy().reshape(56, 56)
denoise_data1 = denoise_data1 / denoise_data1.max()
denoise_data2 = denoise_data2 / denoise_data2.max()

load_pmt2pos = pp.LoadPMT2POS()
clean_pos = load_pmt2pos.reconstruct(denoise_data1, denoise_data2)
noise_pos = load_pmt2pos.reconstruct(data_noise1, data_noise2)
classic_pos = load_pmt2pos.reconstruct(data1, data2)

print('classic_pos:', classic_pos)
print('clean_pos:', clean_pos)
print('noise_pos:', noise_pos)
print('True:', data[index, -3:] / 100)


def plot_3d_data(ax, matrix, color, label):
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    z = matrix
    ax.scatter(x, y, z, color=color, label=label, s=10)


fig = plt.figure(figsize=(14, 8))  # 调整整个图形的大小

# 创建第一个子图
ax1 = fig.add_subplot(121, projection='3d')  # 121表示1行2列的第1个
plot_3d_data(ax1, data1.reshape(56, 56), 'red', "no noise")
plot_3d_data(ax1, data_noise1.reshape(56, 56), 'blue', "noise")
plot_3d_data(ax1, denoise_data1.reshape(56, 56), 'green', "clean")
ax1.set_title(f'PMT 3D Plot - No Noise (index: {index})')
ax1.set_xlabel('X Coordinate')
ax1.set_ylabel('Y Coordinate')
ax1.set_zlabel('Frequency')

# 创建第二个子图
ax2 = fig.add_subplot(122, projection='3d')  # 122表示1行2列的第2个
plot_3d_data(ax2, data2.reshape(56, 56), 'red', "no noise")
plot_3d_data(ax2, data_noise2.reshape(56, 56), 'blue', "noise")
plot_3d_data(ax2, denoise_data2.reshape(56, 56), 'green', "clean")
ax2.set_title(f'PMT 3D Plot - Combined Data (index: {index})')
ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.set_zlabel('Frequency')

# 显示图形
plt.show()

pt = pp.PMTTrans()
pp.PMTHeatmapper(data1.reshape(56, 56), pt.x_array, pt.y_array, f'index = {index}')
