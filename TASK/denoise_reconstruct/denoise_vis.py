from matplotlib import pyplot as plt
import numpy as np
import PyPanda as pp


def plot_3d_data(ax, matrix, color, label):
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    z = matrix
    ax.scatter(x, y, z, color=color, label=label, s=10)


data_noise = np.load(r'../../DATA/Geant4/data_100w.npy')
data_noise = data_noise[0, 56 ** 2:56 ** 2*2].reshape(56, 56)
data_noise = data_noise / data_noise.max()

load_encoder = pp.LoadEncoder()
denoise_data = load_encoder.reconstruct(data_noise)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

matrix1 = denoise_data.cpu().numpy().reshape(56, 56)
matrix1=matrix1*np.sum(data_noise)/np.sum(matrix1)
plot_3d_data(ax, matrix1, 'red', "no noise")  # 使用红色

matrix2 = data_noise
plot_3d_data(ax, matrix2, 'blue', "noise")  # 使用蓝色

ax.set_title('PMT 3D Plot - Combined Data')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Frequency')

ax.legend()
plt.show()
