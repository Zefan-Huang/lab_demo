import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd

def load_data():
    x_train = np.load('output/ready/x_train.npy')
    y_train = np.load('output/ready/y_train.npy')
    x_test = np.load('output/ready/x_test.npy')
    y_test = np.load('output/ready/y_test.npy')

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  #make sure this is one dim and make sure this is float32.
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_loader, test_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 15, out_features=32)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def train_model(train_loader, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    history = []

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print('epoch:{}, loss:{}'.format(epoch, running_loss))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)
                preds = (output>0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
            accuracy = 100 * correct / total
            print('accuracy:{}'.format(accuracy))

        history.append([epoch + 1, avg_loss, accuracy])
    df = pd.DataFrame(history, columns=["epoch", "loss", "accuracy"])
    df.to_csv('result/history.csv', index=False)

    return df

def main():
    train_loader, test_loader = load_data()
    accuracy = train_model(train_loader, test_loader)

if __name__ == '__main__':
    main()