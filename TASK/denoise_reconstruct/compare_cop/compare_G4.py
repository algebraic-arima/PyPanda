import numpy as np
import PyPanda as pp



data = np.load(r'../../../DATA/Geant4/data_15wtest.npy')
load_encoder = pp.LoadEncoder(model_path=r'../../../Model/Encoder/model/encoder2d.pth')
load_pmt2pos = pp.LoadPMT2POS(model_path=r'../../denoise_CNN/model/conv_model_combined_transfer.pth')


G4_pos_list = []
clean_pos_list = []


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

    G4_pos = load_pmt2pos.reconstruct(data1, data2)
    clean_pos = load_pmt2pos.reconstruct(denoise_data1, denoise_data2)

    G4_pos_list.append(G4_pos)
    clean_pos_list.append(clean_pos)


# 将列表转换为numpy数组
G4_pos_array = np.array(G4_pos_list)
clean_pos_array = np.array(clean_pos_list)


# 保存数据到文件
np.save('G4_pos_15w.npy', G4_pos_array)
np.save('G4clean_15w.npy', clean_pos_array)



