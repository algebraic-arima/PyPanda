import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import PyPanda as pt
from torch.utils.tensorboard import SummaryWriter

def RemoveDir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print("remove")
    os.makedirs(log_dir)
    print(f"Created new log directory: {log_dir}")

os.chdir(r'D:\JetBrains\Py Charm Project\PyPanda\Model\Encoder')
RemoveDir(r'.\logs')


class POS2LABEL(Dataset):
    def __init__(self, noisy_data, clean_data):
        self.noisy_data = torch.tensor(noisy_data, dtype=torch.float32)
        self.clean_data = torch.tensor(clean_data, dtype=torch.float32)

    def __len__(self):
        return self.noisy_data.size(0)

    def __getitem__(self, idx):
        # 返回形状为 (1, 56, 56) 的张量
        return self.noisy_data[idx].view(1, 56, 56), self.clean_data[idx].view(1, 56, 56)


class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(ConvDenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128)  # 更新维度，因为输入是 56x56，两次 stride=2 后变为 14x14
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


def add_noise_to_data(data, num_noise_realizations, factor):
    noisy_data_list = []
    for _ in range(num_noise_realizations):
        noise = np.random.normal(0, factor, data.shape)
        noisy_data = data + noise
        noisy_data = np.maximum(noisy_data, 0)
        noisy_data_list.append(noisy_data)
    return np.array(noisy_data_list)


data = np.load("./DATA/data_classic.npy")
data = data[:, :56*56].reshape(-1, 56, 56)  # 将数据转换为 56x56 的矩阵
data = data / data.max()

num_noise_realizations = 2
noisy_data_database = add_noise_to_data(data, num_noise_realizations, 0.1)
data_tile = np.tile(data, (num_noise_realizations, 1, 1))
noisy_data_database_reshape = noisy_data_database.reshape(-1, 56*56)

dataset = POS2LABEL(noisy_data_database_reshape, data_tile.reshape(-1, 56*56))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ConvDenoisingAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 51
writer = SummaryWriter(r".\logs")
for epoch in range(epochs):
    running_loss = 0.0
    for noisy_data, clean_data in dataloader:
        noisy_data = noisy_data.view(-1, 1, 56, 56)  # 调整形状以匹配模型输入
        clean_data = clean_data.view(-1, 1, 56, 56)
        optimizer.zero_grad()
        reconstructed = model(noisy_data)
        loss = criterion(reconstructed, clean_data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    writer.add_scalar("running_loss", torch.log(torch.tensor(running_loss)), epoch)
    writer.flush()

data_noise = np.load(r'D:\JetBrains\Py Charm Project\PyPMT_visualize\data.npy')
data_noise = data_noise[0, 56**2:56**2*2].reshape(56, 56)
data_noise = data_noise / data_noise.max()
pmt_trans = pt.PMTTrans(dis=71)
array = pmt_trans.matrix2array(data_noise)

with torch.no_grad():
    noisy_data = torch.tensor(data_noise, dtype=torch.float32).view(1, 1, 56, 56)  # 调整形状以匹配模型输入
    reconstructed_data = model(noisy_data)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


def plot_3d_data(ax, matrix, color, label):
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    z = matrix
    ax.scatter(x, y, z, color=color, label=label, s=10)


matrix1 =reconstructed_data.numpy().reshape(56,56)
plot_3d_data(ax, matrix1, 'red',"no noise")  # 使用红色

matrix2 = data_noise
plot_3d_data(ax, matrix2, 'blue',"noise")  # 使用蓝色

np.save(r'./save/data_noise.npy',matrix2)
np.save(r'./save/data.npy',matrix1)
torch.save(model.state_dict(), r'./model/encoder2d.pth')

ax.set_title('PMT 3D Plot - Combined Data')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Frequency')

ax.legend()
plt.show()