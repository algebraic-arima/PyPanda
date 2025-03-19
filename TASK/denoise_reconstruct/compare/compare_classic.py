import numpy as np
import PyPanda as pp


def add_noise_to_data(data, data_z, num_noise_realizations,mode):
    noisy_data_list = []
    for i in range(np.size(data,0)):
        z = data_z[i]
        if mode == 'top':
            num_photon = (3.01389248e-02*z**2 + 6.61948458e+01*z + 4.61519056e+04)/100*15
        else:
            num_photon = (8.08150300e-02 * z ** 2 -6.62960040e+01 * z + 3.49185111e+04) /100*15
        for _ in range(num_noise_realizations):
            noise = np.random.poisson(data[i,:,:]/np.sum(data[i,:,:])*num_photon)
            noisy_data_list.append(noise/num_photon*np.sum(data[i,:,:]))
    return np.array(noisy_data_list)


data = np.load(r'../../../DATA/Classic/data_classic_8e3.npy')
dataz=data[:,-1]
load_encoder = pp.LoadEncoder(model_path=r'../../../Model/Encoder/model/encoder2d.pth')
load_pmt2pos = pp.LoadPMT2POS(model_path=r'../../../Model/pmt2pos/model/conv_model_combined.pth')

# 初始化列表来存储重建的pos
classic_pos_list = []
clean_pos_list = []
noise_pos_list = []
print(np.size(data, 0))

for index in range(np.size(data, 0)):
    print(index)

    data1 = data[index, :56 ** 2].reshape(1,56, 56)
    data2 = data[index, 56 ** 2:56 ** 2 * 2].reshape(1,56, 56)
    data1 = data1 / data1.max()
    data2 = data2 / data2.max()

    data_noise1 = add_noise_to_data(data1,dataz, 1,'top').reshape(56, 56)
    data_noise2 = add_noise_to_data(data2,dataz, 1,'bot').reshape(56, 56)

    denoise_data1 = load_encoder.reconstruct(data1).numpy().reshape(56, 56)
    denoise_data2 = load_encoder.reconstruct(data2).numpy().reshape(56, 56)

    denoise_data1 = denoise_data1 / denoise_data1.max()
    denoise_data2 = denoise_data2 / denoise_data2.max()

    classic_pos = load_pmt2pos.reconstruct(data1, data2)
    clean_pos = load_pmt2pos.reconstruct(denoise_data1, denoise_data2)
    noise_pos = load_pmt2pos.reconstruct(data_noise1, data_noise2)

    classic_pos_list.append(classic_pos)
    clean_pos_list.append(clean_pos)
    noise_pos_list.append(noise_pos)


# 将列表转换为numpy数组
classic_pos_array = np.array(classic_pos_list)
clean_pos_array = np.array(clean_pos_list)
noise_pos_array = np.array(noise_pos_list)

# 保存数据到文件
np.save('classic_pos_rescale.npy', classic_pos_array)
np.save('clean_pos_rescale.npy', clean_pos_array)
np.save('noise_pos_rescale.npy', noise_pos_array)
