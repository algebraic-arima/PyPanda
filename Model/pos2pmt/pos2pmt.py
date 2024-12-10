import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor



def RemoveDir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    print(f"Created new log directory: {log_dir}")
    return log_dir


def remove_folders_with_network(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for dir_name in dirs:
            if 'network' in dir_name.lower():
                folder_path = os.path.join(root, dir_name)
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")


def normalize(data, max_val=None):
    if max_val is None:
        max_val = data.max()
    normalized_data = data / max_val
    return normalized_data


class POS2LABEL(Dataset):
    def __init__(self, data, label_idx):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label_idx = label_idx

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        features = self.data[idx, -3:]
        labels = self.data[idx, 0:-3]
        features = features.view(1, 3)
        labels = normalize(labels)
        label = labels[self.label_idx]
        return features, label



class POS2LABELNet(nn.Module):
    def __init__(self):
        super(POS2LABELNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc4(x)
        return x


def train_network(net, train_loader, val_loader, label_idx, epochs=51, log_dir='./logs'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)

    log_dir = os.path.join(log_dir, f"network_{label_idx}")
    writer = SummaryWriter(log_dir=log_dir)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.squeeze()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)

        test_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                outputs = outputs.squeeze()
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
        val_losses.append(test_loss / len(val_loader))
        writer.add_scalar('Validation Loss', test_loss / len(val_loader), epoch)

    writer.close()
    torch.save(net.state_dict(), f'./model/network_{label_idx}.pth')
    return train_losses, val_losses


def main():
    log_dir = RemoveDir(r'\logs')

    all_data = np.load(r'.\DATA\data_classic.npy')
    test_size = 0.2
    train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=42)

    epochs = 51
    all_train_losses = []
    all_val_losses = []

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for label_idx in range(4):
            train_dataset_label = POS2LABEL(train_data, label_idx)
            val_dataset_label = POS2LABEL(val_data, label_idx)
            train_loader_label = DataLoader(train_dataset_label, batch_size=25, drop_last=True, shuffle=True)
            val_loader_label = DataLoader(val_dataset_label, batch_size=25, drop_last=True)
            futures.append(
                executor.submit(train_network, POS2LABELNet(), train_loader_label, val_loader_label, label_idx, epochs,
                                log_dir))

        for future in futures:
            train_losses, val_losses = future.result()
            all_train_losses.extend(train_losses)
            all_val_losses.extend(val_losses)

    writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        train_epoch_losses = [all_train_losses[i * epochs + epoch] for i in range(4)]
        val_epoch_losses = [all_val_losses[i * epochs + epoch] for i in range(4)]

        train_max_loss = max(train_epoch_losses)
        val_max_loss = max(val_epoch_losses)
        train_min_loss = min(train_epoch_losses)
        val_min_loss = min(val_epoch_losses)
        train_avg_loss = np.mean(train_epoch_losses)
        val_avg_loss = np.mean(val_epoch_losses)

        writer.add_scalars('Losses/Training', {
            'Max': train_max_loss,
            'Min': train_min_loss,
            'Avg': train_avg_loss
        }, epoch)

        writer.add_scalars('Losses/Validation', {
            'Max': val_max_loss,
            'Min': val_min_loss,
            'Avg': val_avg_loss
        }, epoch)

    writer.close()
    remove_folders_with_network(log_dir)


if __name__ == '__main__':
    main()
