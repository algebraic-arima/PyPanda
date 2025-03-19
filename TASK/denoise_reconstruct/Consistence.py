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


def test_consistence(data1,data2,mode='clean'):
    data_noise1 = add_noise_to_data(data1, 1).reshape(56, 56)
    data_noise2 = add_noise_to_data(data2, 1).reshape(56, 56)

    load_encoder = pp.LoadEncoder()
    denoise_data1 = load_encoder.reconstruct(data_noise1).numpy().reshape(56, 56)
    denoise_data2 = load_encoder.reconstruct(data_noise2).numpy().reshape(56, 56)

    denoise_data1 = denoise_data1 / denoise_data1.max()
    denoise_data2 = denoise_data2 / denoise_data2.max()

    load_pmt2pos = pp.LoadPMT2POS()
    if mode=='clean':
        reconstruct_pos=load_pmt2pos.reconstruct(denoise_data1, denoise_data2)
    elif mode == 'classic':
        reconstruct_pos=load_pmt2pos.reconstruct(data1, data2)
    else:
        reconstruct_pos=None
    reconstruct_pos=reconstruct_pos.numpy()
    return reconstruct_pos[0][0:2]

index=22
data = np.load(r'../../DATA/Classic/data_classic_8e3.npy')


data1 = data[index, :56 ** 2].reshape(56, 56)
data2 = data[index, 56 ** 2:56 ** 2*2].reshape(56, 56)
data1 = data1 / data1.max()
data2 = data2 / data2.max()


reconstruct_pos=np.zeros((100,2))
classic_pos=test_consistence(data1,data2,mode='classic')
true_pos=data[index,-3:-1]
for i in range(100):
    reconstruct_pos[i,:]=test_consistence(data1,data2)
plt.scatter(reconstruct_pos[:,0],reconstruct_pos[:,1])
plt.scatter(classic_pos[0],classic_pos[1])
plt.scatter(true_pos[0]/100,true_pos[1]/100)
plt.show()