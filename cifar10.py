import torch
import torch.nn as nn
from tqdm import tqdm
from data_cifar10 import *

# torch.manual_seed(24)

device = torch.device('cuda')

EPOCH = 20
LR = 1e-3

channel_size_1 = 16
channel_size_2 = 32

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feat = []
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=channel_size_1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channel_size_1),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_size_1, channel_size_1, 3, 1, 1),
            nn.BatchNorm2d(channel_size_1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_size_1, channel_size_2, 3, 1, 1),
            nn.BatchNorm2d(channel_size_2),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_size_2, channel_size_2, 3, 1, 1),
            nn.BatchNorm2d(channel_size_2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel_size_2, channel_size_2, 3, 1, 1),
            nn.BatchNorm2d(channel_size_2),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(
            #nn.Conv2d(channel_size_2, 10, 1, 1, 0),
            #nn.AvgPool2d(8)
            nn.Flatten(),
            nn.Linear(channel_size_2 * 8 * 8, 10)
        )
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

    def forward(self, x):
        self.feat.clear()
        for layer in self.conv_layers:
            x = layer(x)
            self.feat.append(x.view(x.size(0), -1))
        output = self.out(x).view(x.size(0), 10)
        return output

    def forward_to(self, x, to):
        for layer in self.conv_layers[:to]:
            x = layer(x)
        return x

if __name__ == '__main__':
    cnn = CNN()
    cnn.to(device)
    cnn.train()
    # print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()
    running_correct = 0

    # print(train_data[0][0].size())
    # print(len(train_loader))
    
    for epoch in range(EPOCH):
        running_loss = 0
        for b_x, b_y in tqdm(train_loader, desc = 'Epoch {:d}/{:d}'.format(epoch, EPOCH)):
            output = cnn(b_x.cuda())
            loss = loss_func(output, b_y.cuda())
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.clone().detach().cpu().numpy()
            optimizer.step()
            
        print('Train Loss: {:.5f}'.format(running_loss / len(train_loader)))

        cnn.eval()
        with torch.no_grad():
            training_correct = 0
            for X_train, y_train in train_loader:
                X_train = X_train.to(device)
                y_train = y_train.to(device)

                outputs = cnn(X_train)
                y_pred = torch.max(outputs, 1).indices
                training_correct += torch.sum(y_pred == y_train).item()

            testing_correct = 0
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)

                outputs = cnn(X_test)
                y_pred = torch.max(outputs, 1).indices
                testing_correct += torch.sum(y_pred == y_test).item()

        print('Train Acc {:.4f}%'.format(training_correct / len(train_data) * 100))
        print('Test Acc {:.4f}%'.format(testing_correct / len(test_data) * 100))

    torch.save(cnn.state_dict(), './models/CIFAR10_no_dropout/model_2.pt')
