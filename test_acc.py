import torch
from data_cifar10 import *

device = torch.device('cuda')

def test_acc(model):
    model.eval()
    with torch.no_grad():
        training_correct = 0
        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            outputs = model(X_train)
            y_pred = torch.max(outputs, 1).indices
            training_correct += torch.sum(y_pred == y_train).item()

        testing_correct = 0
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            outputs = model(X_test)
            y_pred = torch.max(outputs, 1).indices
            testing_correct += torch.sum(y_pred == y_test).item()

    return (training_correct / len(train_data)), \
        (testing_correct / len(test_data))

def print_acc(model, name):
    a1, a2 = test_acc(model)
    print('{:s} Train Acc {:.4f}%'.format(name, a1 * 100))
    print('{:s} Test Acc {:.4f}%'.format(name, a2 * 100))
    
