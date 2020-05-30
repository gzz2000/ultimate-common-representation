import torch
import torch.nn as nn
from tqdm import trange
from data_cifar10 import *
from cka import each_layer_pair_cka
from cifar10 import CNN

torch.manual_seed(721)
device = torch.device('cuda')

LR = 0.001

class Converter(CNN):
    def __init__(self, models, input_id, output_id):
        super(Converter, self).__init__()
        self.models = models
        self.input_id = input_id
        self.output_id = output_id
        self.param = [param for layer in self.conv_layers[input_id:output_id] for param in layer.parameters()]

    def forward(self, x):
        self.feat.clear()
        for layer in self.models[0].conv_layers[:self.input_id]:
            x = layer(x)
            self.feat.append(x.view(x.size(0), -1))
        for i in range(self.input_id, self.output_id):
            x = self.conv_layers[i](x)
            self.feat.append(x.view(x.size(0), -1))
        for layer in self.models[1].conv_layers[self.output_id:]:
            x = layer(x)
            self.feat.append(x.view(x.size(0), -1))
        return self.models[1].out(x).view(x.size(0), 10)

def test(cnn, mu = 0, fg = False, pro = None, pos = None):
    loss_f = nn.CrossEntropyLoss()
    cnn.eval()
    if(fg == True):
        pro.eval()
    test_correct = 0
    test_loss = 0
    train_correct = 0
    train_loss = 0
    with torch.no_grad():

        for x_train, y_train in train_loader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            outputs = cnn(x_train)
            if(fg == True):
                pro(x_train)
                loss = (1-mu)*loss_f(outputs, y_train)+mu * \
                       torch.dist(cnn.feat[pos], pro.feat[pos],
                                  p=2)
            else:
                loss = loss_f(outputs, y_train)
            y_pred = torch.max(outputs, 1).indices

            train_correct += torch.sum(y_pred == y_train).item()

            train_loss += loss.item()

        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            outputs = cnn(x_test)
            if(fg == True):
                pro(x_test)
                loss = (1-mu)*loss_f(outputs, y_test)+mu * \
                       torch.dist(cnn.feat[pos], pro.feat[pos],
                                  p=2)
            else:
                loss = loss_f(outputs, y_test)
            y_pred = torch.max(outputs, 1).indices

            test_correct += torch.sum(y_pred == y_test).item()
            test_loss += loss.item()
    print('train loss {:.4f}'.format(train_loss / len(train_loader)))
    print('test loss {:.4f}'.format(test_loss / len(test_loader)))

    print('train acc {:.4f}%'.format(train_correct / len(train_data) * 100))
    print('test acc {:.4f}%'.format(test_correct / len(test_data) * 100))

def adv_train(cnn1, cnn2, cnn, pos = None):
    cnn1.train()
    cnn2.train()
    cnn.train()

    opt1 = torch.optim.Adam(cnn1.parameters(), LR)
    opt2 = torch.optim.Adam(cnn2.parameters(), LR)
    opt = torch.optim.Adam(cnn.param, LR)

    loss_f = nn.CrossEntropyLoss()
    EPOCH = 30
    iteration = trange(EPOCH)
    for epoch in iteration:
        running_loss = 0
        running_loss_1 = 0
        running_loss_2 = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            output1 = cnn1(x)
            output2 = cnn2(x)
            output = cnn(x)

            loss = loss_f(output, y)
            running_loss += loss.item()
            loss1 = loss_f(output1, y)
            running_loss_1 += loss1.item()
            loss1 -= loss / 2
            loss2 = loss_f(output2, y)
            running_loss_2 += loss2.item()
            #loss2 -= loss

            opt1.zero_grad()
            loss1.backward(retain_graph=True)
            opt1.step()
            opt2.zero_grad()
            loss2.backward(retain_graph=True)
            opt2.step()
            opt.zero_grad()
            loss.backward()
            opt.step()
        iteration.set_description("cnn1:%.2f cnn2:%.2f cnn:%.2f" %
                                  (running_loss_1 / len(train_loader), running_loss_2 / len(train_loader), running_loss / len(train_loader)))

if __name__ == '__main__':
    cnn_1 = CNN().to(device)
    cnn_2 = CNN().to(device)
    cnn_1.train()
    cnn_2.train()
    cnn = Converter([cnn_1, cnn_2], 1, 3).to(device)

    adv_train(cnn_1, cnn_2, cnn)
    test(cnn_1)
    test(cnn_2)
    test(cnn)
    torch.save(cnn_1.state_dict(), 'model_a.pt')
    torch.save(cnn_1.state_dict(), 'model_b.pt')

