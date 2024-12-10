import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import PyPanda as pt


class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(ConvDenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 32 * 14 * 14),
            nn.ReLU(),
            nn.Unflatten(1, (32, 14, 14)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


data_noise = np.load(r'../../DATA/Geant4/data_100w.npy')
data_noise = data_noise[0, 56 ** 2:56 ** 2*2].reshape(56, 56)
data_noise = data_noise / data_noise.max()
pmt_trans = pt.PMTTrans(dis=71)
array = pmt_trans.matrix2array(data_noise)
model = ConvDenoisingAutoencoder()
model.load_state_dict(torch.load(r'../../Model/Encoder/model/encoder2d.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))
with torch.no_grad():
    noisy_data = torch.tensor(data_noise, dtype=torch.float32).view(1, 1, 56, 56)  # 调整形状以匹配模型输入
    reconstructed_data = model(noisy_data)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


def plot_3d_data(ax, matrix, color, label):
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    z = matrix
    ax.scatter(x, y, z, color=color, label=label, s=10)


matrix1 = reconstructed_data.numpy().reshape(56, 56)
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
