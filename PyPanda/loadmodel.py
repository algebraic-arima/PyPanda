import torch
import torch.nn as nn


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


class LoadEncoder:
    def __init__(self, model_path=r'../../Model/Encoder/model/encoder2d.pth'):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ConvDenoisingAutoencoder().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.model.eval()

    def reconstruct(self, noisy_data):
        noisy_data=torch.tensor(noisy_data, dtype=torch.float32).view(1, 1, 56, 56)
        with torch.no_grad():
            reconstructed_data = self.model(noisy_data)
        return reconstructed_data


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


class LoadPMT2POS:
    def __init__(self, model_path=r'../../Model/pmt2pos/model/conv_model_combined.pth'):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CombinedCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.model.eval()

    def reconstruct(self, matrix_top,matrix_bot):
        matrix_top=torch.tensor(matrix_top, dtype=torch.float32).view(1,56, 56)
        matrix_bot = torch.tensor(matrix_bot, dtype=torch.float32).view(1,56, 56)
        with torch.no_grad():
            pos = self.model(matrix_top,matrix_bot)
        return pos