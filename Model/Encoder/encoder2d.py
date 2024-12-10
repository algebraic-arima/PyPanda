import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def RemoveDir(log_dir):
    if os.path.exists(log_dir):
        file_list = os.listdir(log_dir)
        for file in file_list:
            file_path = os.path.join(log_dir, file)
            os.remove(file_path)
    else:
        os.makedirs(log_dir)
    print(f"Created new log directory: {log_dir}")


class POS2LABEL(Dataset):
    def __init__(self, noisy_data, clean_data):
        self.noisy_data = torch.tensor(noisy_data, dtype=torch.float32)
        self.clean_data = torch.tensor(clean_data, dtype=torch.float32)

    def __len__(self):
        return self.noisy_data.size(0)

    def __getitem__(self, idx):
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


def add_noise_to_data(data, num_noise_realizations, factor):
    noisy_data_list = []
    for _ in range(num_noise_realizations):
        noise = np.random.normal(0, factor, data.shape)
        noisy_data = data + noise
        noisy_data = np.maximum(noisy_data, 0)
        noisy_data_list.append(noisy_data)
    return np.array(noisy_data_list)


RemoveDir(r'./logs')

data = np.load("./DATA/data_classic.npy")
data = data[:, :56 * 56].reshape(-1, 56, 56)  # 将数据转换为 56x56 的矩阵
data = data / data.max()

num_noise_realizations = 10
factor=0.1
noisy_data_database = add_noise_to_data(data, num_noise_realizations, factor)
data_tile = np.tile(data, (num_noise_realizations, 1, 1))
noisy_data_database_reshape = noisy_data_database.reshape(-1, 56 * 56)

dataset = POS2LABEL(noisy_data_database_reshape, data_tile.reshape(-1, 56 * 56))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
model = ConvDenoisingAutoencoder().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 51
writer = SummaryWriter(r"./logs")
for epoch in range(epochs):
    running_loss = 0.0
    for noisy_data, clean_data in dataloader:
        noisy_data = noisy_data.view(-1, 1, 56, 56).to(device=device) # 调整形状以匹配模型输入
        clean_data = clean_data.view(-1, 1, 56, 56).to(device=device)
        optimizer.zero_grad()
        reconstructed = model(noisy_data)
        loss = criterion(reconstructed, clean_data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        print(f'epoch: {epoch}\nrunning_loss={running_loss}')
    writer.add_scalar("running_loss", torch.log(torch.tensor(running_loss)), epoch)
    writer.flush()


torch.save(model.state_dict(), r'./model/encoder2d.pth')
