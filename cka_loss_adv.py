import torch
from torch import nn
from cifar10 import device, CNN, train_data, train_loader, test_loader, test_data
from tqdm import trange, tqdm

torch.manual_seed(23)

cnn1 = CNN()
cnn1.to(device)
cnn1.eval()
cnn2 = CNN()
cnn2.to(device)
cnn2.train()

cnn1.load_state_dict(torch.load('model_1.pt'))
LR = 1e-3
EPOCH = 30
cka_coef = 0.5

def feature_space_linear_cka(X, Y):
	X = X - torch.mean(X, 0, keepdims=True)
	Y = Y - torch.mean(Y, 0, keepdims=True)
	dot_sim = X.T.matmul(Y).norm() ** 2
	norm_x = X.T.matmul(X).norm()
	norm_y = Y.T.matmul(Y).norm()
	return dot_sim / (norm_x * norm_y)

optimizer = torch.optim.Adam(cnn2.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
running_correct = 0

print(train_data[0][0].size())
print(len(train_loader))
for epoch in range(EPOCH):
	running_loss = 0
	iteration = tqdm(train_loader)
	for x, y in iteration:
		with torch.no_grad():
			cnn1(x.to(device))
		output = cnn2(x.to(device))

		cka_loss = 0
		for f1, f2 in zip(cnn1.feat, cnn2.feat):
			cka_loss += feature_space_linear_cka(f1, f2)
		cka_loss /= len(cnn1.feat)
		loss = loss_func(output, y.to(device)) + cka_loss * cka_coef
		optimizer.zero_grad()
		loss.backward()
		running_loss += loss.clone().detach().cpu().numpy()
		optimizer.step()
		iteration.set_description('Epoch %d:%.3f %.3f' % (epoch + 1, loss, cka_loss * cka_coef))
torch.save(cnn2.state_dict(), 'model_3.pt')

with torch.no_grad():
	training_correct = 0
	for X_train, y_train in train_loader:
		X_train = X_train.to(device)
		y_train = y_train.to(device)

		outputs = cnn2(X_train)
		y_pred = torch.max(outputs, 1).indices
		training_correct += torch.sum(y_pred == y_train).item()

	testing_correct = 0
	for X_test, y_test in test_loader:
		X_test = X_test.to(device)
		y_test = y_test.to(device)

		outputs = cnn2(X_test)
		y_pred = torch.max(outputs, 1).indices
		testing_correct += torch.sum(y_pred == y_test).item()

print('Train Acc {:.4f}%'.format(training_correct / len(train_data) * 100))
print('Test Acc {:.4f}%'.format(testing_correct / len(test_data) * 100))
