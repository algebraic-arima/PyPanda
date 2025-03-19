from matplotlib import pyplot as plt
import numpy as np
import PyPanda as pp


def add_noise_to_data(data, num_noise_realizations):
    noisy_data_list = []
    for _ in range(num_noise_realizations):
        noise = np.random.poisson(data*30000/np.sum(data))
        noisy_data = data + noise
        noisy_data = np.maximum(noisy_data, 0)
        noisy_data_list.append(noisy_data/30000*np.sum(data))
    return np.array(noisy_data_list)


def test_invariant(data1,data2):
    data_noise1 = add_noise_to_data(data1, 1).reshape(56, 56)
    data_noise2 = add_noise_to_data(data2, 1).reshape(56, 56)

    load_encoder = pp.LoadEncoder()
    denoise_data1 = load_encoder.reconstruct(data_noise1).numpy().reshape(56, 56)
    denoise_data2 = load_encoder.reconstruct(data_noise2).numpy().reshape(56, 56)

    denoise_data1 = denoise_data1 / denoise_data1.max()
    denoise_data2 = denoise_data2 / denoise_data2.max()

    return denoise_data1,denoise_data2


def calculate_error(data1,data2,denoise_data1,denoise_data2):
    error1=sum(sum((denoise_data1-data1)**2))
    error2 = sum(sum((denoise_data2 - data2) ** 2))
    return error1,error2



index=0
data = np.load(r'../../DATA/Classic/data_classic_8e3.npy')


data1 = data[index, :56 ** 2].reshape(56, 56)
data2 = data[index, 56 ** 2:56 ** 2*2].reshape(56, 56)
data1 = data1 / data1.max()
data2 = data2 / data2.max()
data01=data1
data02=data2
error1=np.zeros(100)
error2=np.zeros(100)
for i in range(100):
    denoise_data1,denoise_data2=test_invariant(data1,data2)
    error1[i],error2[i]=calculate_error(data01,data02,denoise_data1,denoise_data2)
    data1=denoise_data1
    data2=denoise_data2


plt.plot(range(100),error1)
plt.plot(range(100),error2)
plt.show()



def plot_3d_data(ax, matrix, color, label):
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    z = matrix
    ax.scatter(x, y, z, color=color, label=label, s=10)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# matrix1=matrix1*np.sum(data_noise)/np.sum(matrix1)
plot_3d_data(ax, data01, 'red', "no noise")  # 使用红色


plot_3d_data(ax, denoise_data1, 'green', "clean")  # 使用红色


ax.set_title(f'PMT 3D Plot - Combined Data for index {index}')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Frequency')

ax.legend()
plt.show()