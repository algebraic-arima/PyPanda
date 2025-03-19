import PyPanda as pp
import numpy as np

data = np.load(r'DATA/Geant4/data_100w.npy')
index = 5
print(data.shape)
data1 = data[index, :56 ** 2].reshape(56, 56)
data2 = data[index, 56 ** 2:56 ** 2 * 2].reshape(56, 56)
[x, y, z] = data[index, -3:]

pt = pp.trans.PMTTrans()

# pp.PMTHeatmapper(pt.mask, pt.x_array, pt.y_array, "mask")
pp.PMTHeatmapper(data1, pt.x_array, pt.y_array, f"for row data, index = {index}, upper PMT, pos = ({x}, {y}, {z})")
pp.PMTHeatmapper(data2, pt.x_array, pt.y_array, f"for row data, index = {index}, lower PMT, pos = ({x}, {y}, {z})")

data_clean = np.load(r'DATA/Classic/data_classic.npy')
index = 6
print(data_clean.shape)
data1 = data_clean[index, :56 ** 2].reshape(56, 56)
data2 = data_clean[index, 56 ** 2:56 ** 2 * 2].reshape(56, 56)
[x, y, z] = data_clean[index, -3:]
pp.PMTHeatmapper(data1.T, pt.x_array, pt.y_array, f"for classic data, index = {index}, upper PMT, pos = ({x}, {y}, {z})")
pp.PMTHeatmapper(data2.T, pt.x_array, pt.y_array, f"for classic data, index = {index}, lower PMT, pos = ({x}, {y}, {z})")

