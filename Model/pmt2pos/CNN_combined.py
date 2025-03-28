import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split

from PyPanda.loadmodel import CombinedCNN


def RemoveDir(log_dir):
    if os.path.exists(log_dir):
        file_list = os.listdir(log_dir)
        for file in file_list:
            file_path = os.path.join(log_dir, file)
            os.remove(file_path)
    else:
        os.makedirs(log_dir)
    print(f"Created new log directory: {log_dir}")


def normalize(data, max_val=None):
    if max_val is None:
        max_val = data.max()
    normalized_data = data / max_val
    return normalized_data


class MYDATA(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        features1 = self.data[idx, 0:56 * 56]
        features2 = self.data[idx, 56 * 56 :56 * 56 * 2 ]
        features1 = normalize(features1).view(1, 56, 56)
        features2 = normalize(features2).view(1, 56, 56)
        label = normalize(self.data[idx, -3:],100)
        return features1, features2, label


RemoveDir(r'.\conv_logs_combined')

all_data = np.load("../../DATA/Classic/data_classic_8e3.npy")
test_size = 0.1
train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=42)
train_dataset = MYDATA(train_data)
val_dataset = MYDATA(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=64, drop_last=True, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, drop_last=True)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CombinedCNN().to(device)
print(model)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 301

writer = SummaryWriter(r".\conv_logs_combined")

for epoch in range(epochs):
    running_loss = 0.0
    for data in train_dataloader:
        inputs, matrix_sum, labels = data
        inputs, matrix_sum, labels = inputs.to(device), matrix_sum.to(device), labels.to(device)
        labels = labels[:, 0:model.num_labels]
        outputs = model(inputs, matrix_sum)
        result = loss(outputs, labels)
        optimizer.zero_grad()
        result.backward()
        optimizer.step()
        running_loss += result.item()
    if epoch % 5 == 0:
        print(f'epoch: {epoch}\nrunning_loss={running_loss}')
    writer.add_scalar("running_loss", torch.log(torch.tensor(running_loss)), epoch)

    test_loss = 0.0
    for data in val_dataloader:
        inputs, matrix_sum, labels = data
        inputs, matrix_sum, labels = inputs.to(device), matrix_sum.to(device), labels.to(device)
        labels = labels[:, 0:model.num_labels]
        outputs = model(inputs, matrix_sum)
        result = loss(outputs, labels)
        test_loss += result.item()
    if epoch % 5 == 0:
        print(f'test_loss={test_loss}')
    writer.add_scalar("test_loss", torch.log(torch.tensor(test_loss)), epoch)
    writer.flush()

torch.save(model.state_dict(),  r'./model/conv_model_combined.pth')
