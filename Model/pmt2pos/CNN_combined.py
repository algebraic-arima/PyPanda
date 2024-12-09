import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import os
import shutil
from sklearn.model_selection import train_test_split

# Please modify os.chdir to your specific path
current_dir = r'D:\JetBrains\Py Charm Project\CNN4Pandax_new'
os.chdir(r'D:\JetBrains\Py Charm Project\CNN4Pandax_new')


def RemoveDir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    print(f"Created new log directory: {log_dir}")


RemoveDir(current_dir + r'\conv_logs_combined')

all_data = np.load(r'.\DATA\data_classic.npy')

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


test_size = 0.2
train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=42)
train_dataset = MYDATA(train_data)
val_dataset = MYDATA(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=25, drop_last=True, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=25, drop_last=True)


class CombinedCNN(nn.Module):
    def __init__(self, num_labels=3):
        super(CombinedCNN, self).__init__()

        # 56x56输入的卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32个过滤器，每个3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 56x56输入的卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32个过滤器，每个3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(14 * 14 * 64 + 14 * 14 * 64, 128)  # 全连接层
        self.fc2 = nn.Linear(128, num_labels)  # 输出层，对应3个标签
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1 = x1.view(-1, 14 * 14 * 64)
        x2 = x2.view(-1, 14 * 14 * 64)
        x = torch.cat((x1, x2), 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CombinedCNN().to(device)
print(model)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 101

writer = SummaryWriter(current_dir + r"\conv_logs_combined")

for epoch in range(epochs):
    running_loss = 0.0
    for data in train_dataloader:
        inputs, matrix_sum, labels = data
        inputs, matrix_sum, labels = inputs.to(device), matrix_sum.to(device), labels.to(device)
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
        outputs = model(inputs, matrix_sum)
        errors = labels - outputs
        errors = errors.detach().numpy()
        for i in range(errors.shape[1]):
            writer.add_histogram(f"val_error_{i}", errors[:, i], epoch)
        result = loss(outputs, labels)
        test_loss += result.item()
    if epoch % 5 == 0:
        print(f'test_loss={test_loss}')
    writer.add_scalar("test_loss", torch.log(torch.tensor(test_loss)), epoch)
    writer.flush()

torch.save(model.state_dict(), current_dir + r'./model/conv_model_combined.pth')
