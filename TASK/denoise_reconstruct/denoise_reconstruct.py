from matplotlib import pyplot as plt
import numpy as np
import PyPanda as pp


index=4
# data = np.load(r'../../DATA/Classic/data_classic.npy')
data = np.load(r'../../DATA/Geant4/data_100w.npy')
data1 = data[index, :56 ** 2].reshape(56, 56)
data2 = data[index, 56 ** 2:56 ** 2*2].reshape(56, 56)
data1 = data1 / data1.max()
data2 = data2 / data2.max()
data_noise1=data1*(1+np.random.normal(0, 0.1, data1.shape))
data_noise2=data2*(1+np.random.normal(0, 0.1, data2.shape))
data_noise1 = data_noise1 / data_noise1.max()
data_noise2 = data_noise2 / data_noise2.max()

load_encoder = pp.LoadEncoder()
denoise_data1 = load_encoder.reconstruct(data_noise1).numpy().reshape(56, 56)
denoise_data2 = load_encoder.reconstruct(data_noise2).numpy().reshape(56, 56)
denoise_data1=denoise_data1/denoise_data1.max()
denoise_data2=denoise_data2/denoise_data2.max()

load_pmt2pos = pp.LoadPMT2POS()
reconstructed_pos = load_pmt2pos.reconstruct(denoise_data1,denoise_data2)
raw_pos = load_pmt2pos.reconstruct(data_noise1,data_noise2)

print('reconstructed_pos:',reconstructed_pos)
print('raw_pos:',raw_pos)
print('True:',data[index,-3:]/100)

def plot_3d_data(ax, matrix, color, label):
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    z = matrix
    ax.scatter(x, y, z, color=color, label=label, s=10)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# matrix1=matrix1*np.sum(data_noise)/np.sum(matrix1)
plot_3d_data(ax, data1, 'red', "no noise")  # 使用红色

# matrix2 = data_noise
plot_3d_data(ax, data_noise1, 'blue', "noise")  # 使用蓝色

plot_3d_data(ax, denoise_data1, 'green', "clean")  # 使用红色


ax.set_title('PMT 3D Plot - Combined Data')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Frequency')

ax.legend()
plt.show()


