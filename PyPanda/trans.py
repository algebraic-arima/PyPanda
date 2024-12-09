import numpy as np
import pandas as pd



class PMTTrans:
    def __init__(self, dis):
        self.dis_top = dis
        self.coordinates = self.generate_coordinates()

    def generate_coordinates(self):
        x_coords, y_coords = [], []
        for i in range(100):
            for j in range(100):
                x = -50 * self.dis_top + self.dis_top / 2 + i * self.dis_top
                y = -50 * self.dis_top + self.dis_top / 2 + j * self.dis_top
                if x * x + y * y < 960 * 960:
                    x_coords.append(x - 12.125)
                    y_coords.append(y + 12.125)

                    x_coords.append(x + 12.125)
                    y_coords.append(y + 12.125)

                    x_coords.append(x - 12.125)
                    y_coords.append(y - 12.125)

                    x_coords.append(x + 12.125)
                    y_coords.append(y - 12.125)
        return list(zip(x_coords, y_coords))

    def array2matrix(self, array):
        all_pmt_numbers=range(2304)
        count_dict = {pmt_num: 0 for pmt_num in all_pmt_numbers}
        for index, value in enumerate(array):
            count_dict[index] = value
        n_counts = np.array(list(count_dict.values()))

        df = pd.DataFrame(self.coordinates, columns=['X', 'Y'])
        unique_x = sorted(df['X'].unique())
        unique_y = sorted(df['Y'].unique())

        freq_matrix = np.zeros((len(unique_y), len(unique_x)))
        for idx, (x, y) in enumerate(self.coordinates):
            ix = unique_x.index(x)
            iy = unique_y.index(y)
            freq_matrix[iy, ix] = n_counts[idx]
        return freq_matrix

    def matrix2array(self, freq_matrix):
        # 假设freq_matrix是一个二维numpy数组
        # 获取唯一的x和y坐标值
        df = pd.DataFrame(self.coordinates, columns=['X', 'Y'])
        unique_x = sorted(df['X'].unique())
        unique_y = sorted(df['Y'].unique())

        # 初始化一个空数组，用于存储结果
        array = np.zeros((2304,))

        # 遍历freq_matrix，并更新array
        for idx, (x, y) in enumerate(self.coordinates):
            ix = unique_x.index(x)
            iy = unique_y.index(y)
            array[idx] = freq_matrix[iy, ix]

        return array

    def create_mask(self):
        all_pmt_numbers = range(2304)
        count_dict = {pmt_num: 0 for pmt_num in all_pmt_numbers}
        for index, value in enumerate(np.ones(2304)):
            count_dict[index] = value
        n_counts = np.array(list(count_dict.values()))

        df = pd.DataFrame(self.coordinates, columns=['X', 'Y'])
        unique_x = sorted(df['X'].unique())
        unique_y = sorted(df['Y'].unique())

        freq_matrix = np.zeros((len(unique_y), len(unique_x)))
        for idx, (x, y) in enumerate(self.coordinates):
            ix = unique_x.index(x)
            iy = unique_y.index(y)
            freq_matrix[iy, ix] = n_counts[idx]
        return freq_matrix
