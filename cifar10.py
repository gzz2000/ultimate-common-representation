import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from tqdm import trange

torch.manual_seed(23)
device = torch.device('cuda')

EPOCH = 20
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.CIFAR10(
	root='./data',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.feat = []
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=16,
				kernel_size=5,
				stride=1,
				padding=2,
			),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.out = nn.Linear(32*4*4, 10)
		self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

	def forward(self, x):
		self.feat.clear()
		for layer in self.conv_layers:
			x = layer(x)
			self.feat.append(x.view(x.size(0), -1))
		x = x.view(x.size(0), -1)
		output = self.out(x)
		return output

if __name__ == '__main__':
	cnn = CNN()
	cnn.to(device)
	print(cnn)

	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
	loss_func = nn.CrossEntropyLoss()
	running_correct = 0

	print(train_data[0][0].size())
	print(len(train_loader))
	iteration = trange(EPOCH)
	for epoch in iteration:
		running_loss = 0
		for step, (b_x, b_y) in enumerate(train_loader):
			output = cnn(b_x.cuda())
			loss = loss_func(output, b_y.cuda())
			optimizer.zero_grad()
			loss.backward()
			running_loss += loss.clone().detach().cpu().numpy()
			optimizer.step()
		iteration.set_description(str(running_loss / len(train_loader)))
	torch.save(cnn.state_dict(), 'model2.pt')

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
