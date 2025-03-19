import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import PyPanda as pp


class POS2LABEL(Dataset):
    def __init__(self, noisy_data, clean_data):
        self.noisy_data = torch.tensor(noisy_data, dtype=torch.float32)
        self.clean_data = torch.tensor(clean_data, dtype=torch.float32)

    def __len__(self):
        return self.noisy_data.size(0)

    def __getitem__(self, idx):
        clean_data=self.clean_data[idx]
        max=clean_data.max()
        return (self.noisy_data[idx]/max).view(1, 56, 56), (self.clean_data[idx]/max).view(1, 56, 56)


class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(ConvDenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*14*14, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 32*14*14),
            nn.ReLU(),
            nn.Unflatten(1, (32, 14, 14)),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0, output_padding=0),
        )

    def forward(self, x ,mask):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded_masked=decoded*mask
        return decoded_masked


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


data = np.load("../../DATA/Classic/data_classic_8e3.npy")
data_z = data[:, -1]
data = data[:, :56 * 56].reshape(-1, 56, 56)

num_noise_realizations = 3
noisy_data_database = add_noise_to_data(data,data_z, num_noise_realizations,'top')
data_tile_top = np.tile(data, (num_noise_realizations, 1, 1))
noisy_data_database_reshape_top = noisy_data_database.reshape(-1, 56 * 56)

data = data[:, 56 * 56:56**2*2].reshape(-1, 56, 56)

noisy_data_database = add_noise_to_data(data,data_z, num_noise_realizations,'bot')
data_tile_bot = np.tile(data, (num_noise_realizations, 1, 1))
noisy_data_database_reshape_bot = noisy_data_database.reshape(-1, 56 * 56)

noisy_data_database_reshape=np.concatenate((noisy_data_database_reshape_top,noisy_data_database_reshape_bot),axis=0)
data_tile=np.concatenate((data_tile_top,data_tile_bot),axis=0)

dataset = POS2LABEL(noisy_data_database_reshape, data_tile.reshape(-1, 56 * 56))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
model = ConvDenoisingAutoencoder().to(device)
model_path = r'./model/encoder2d.pth'
model.load_state_dict(torch.load(model_path))
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
ppp=pp.PMTTrans(71)
mask=ppp.create_mask()
mask=torch.tensor(mask,dtype=torch.float32).to(device)

start_epoch = 102
epochs = 100

writer = SummaryWriter(r"./logs")
for epoch in range(start_epoch, start_epoch + epochs):
    running_loss = 0.0
    for noisy_data, clean_data in dataloader:
        noisy_data = noisy_data.view(-1, 1, 56, 56).to(device=device) # 调整形状以匹配模型输入
        clean_data = clean_data.view(-1, 1, 56, 56).to(device=device)
        optimizer.zero_grad()
        reconstructed = model(noisy_data,mask)
        loss = criterion(reconstructed, clean_data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        print(f'epoch: {epoch}\nrunning_loss={running_loss/num_noise_realizations}')
    writer.add_scalar("running_loss", torch.log(torch.tensor(running_loss/num_noise_realizations)), epoch)
    writer.flush()


torch.save(model.state_dict(), r'./model/encoder2d_con.pth')
