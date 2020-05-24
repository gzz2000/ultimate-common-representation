import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from tqdm import trange

torch.manual_seed(721)
device = torch.device('cuda')

EPOCH = 5
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.CIFAR10(
	root='./data',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.CIFAR10(
	root='./data', train=False, transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(
	dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.feat = []
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=32,
				kernel_size=3,
				stride=1,
				padding=1,
			),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 96, 3, 1, 1),
			nn.BatchNorm2d(96),
			nn.ReLU(True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(96, 96, 3, 1, 1),
			nn.BatchNorm2d(96),
			nn.ReLU(True),
			nn.MaxPool2d(2, 2)
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(96, 96, 3, 1, 1),
			nn.BatchNorm2d(96),
			nn.ReLU(True),
		)
		self.out = nn.Sequential(
			nn.Flatten(),
			nn.Linear(96 * 8 * 8, 10)
		)
		self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

	def forward(self, x):
		self.feat.clear()
		for layer in self.conv_layers:
			x = layer(x)
			self.feat.append(x.view(x.size(0), -1))
		output = self.out(x).view(x.size(0), 10)
		return output

def calc(cnn, mu, Str, fg,  pro=None, pos=None):
	pos = 1
	cnn.to(device)
	if(fg == True):
		pro.to(device)
		pro.train()
	cnn.train()

	optimizer = torch.optim.Adam(
		filter(lambda q: q.requires_grad, cnn.parameters()), lr=LR)

	loss_f = nn.CrossEntropyLoss()

	#    print(train_data[0][0].size())
	#    print(len(train_loader))

	iteration = trange(EPOCH)
	for epoch in iteration:
		running_correct = 0
		running_loss = 0
		for step, (b_x, b_y) in enumerate(train_loader):
			b_x = b_x.to(device)
			b_y = b_y.to(device)
			outputs = cnn(b_x)
			if(fg == True):
				with torch.no_grad():
					pro(b_x)
				loss = (1-mu)*loss_f(outputs, b_y)+mu * \
					   torch.dist(cnn.feat[pos], pro.feat[pos], p=2) / len(cnn.feat[pos])
			else:
				loss = loss_f(outputs, b_y)

			optimizer.zero_grad()
			y_pred = torch.max(outputs, 1).indices

			loss.backward()
			running_loss += loss.item()
			running_correct += torch.sum(y_pred == b_y).item()
			optimizer.step()
		iteration.set_description(str(running_loss / len(train_loader)))

	#    if(fg == False):
	#torch.save(cnn.state_dict(), 'model_{}.pt'.format(Str))

	cnn.eval()
	if(fg == True):
		pro.eval()
	test_correct = 0
	test_loss = 0
	train_correct = 0
	train_loss = 0
	with torch.no_grad():

		for X_train, y_train in train_loader:
			X_train = X_train.to(device)
			y_train = y_train.to(device)

			outputs = cnn(X_train)
			if(fg == True):
				pro(X_train)
				loss = (1-mu)*loss_f(outputs, y_train)+mu * \
					   torch.dist(cnn.feat[pos], pro.feat[pos], p=2) / len(cnn.feat[pos])
			else:
				loss = loss_f(outputs, y_train)
			y_pred = torch.max(outputs, 1).indices

			train_correct += torch.sum(y_pred == y_train).item()

			train_loss += loss.item()

		for X_test, y_test in test_loader:
			X_test = X_test.to(device)
			y_test = y_test.to(device)

			outputs = cnn(X_test)
			if(fg == True):
				pro(X_test)
				loss = (1-mu)*loss_f(outputs, y_test)+mu * \
					   torch.dist(cnn.feat[pos], pro.feat[pos], p=2) / len(cnn.feat[pos])
			else:
				loss = loss_f(outputs, y_test)
			y_pred = torch.max(outputs, 1).indices

			test_correct += torch.sum(y_pred == y_test).item()
			test_loss += loss.item()
	print(Str)
	print('Train Loss {:.4f}'.format(train_loss / len(train_loader)))
	print('Test Loss {:.4f}'.format(test_loss / len(test_loader)))

	print('Train Acc {:.4f}%'.format(train_correct / len(train_data) * 100))
	print('Test Acc {:.4f}%'.format(test_correct / len(test_data) * 100))


if __name__ == '__main__':
	cnn_1 = CNN()
	cnn_2 = CNN()

	#    calc(cnn_1, 0, '1', False)
	#    calc(cnn_2, 0, '2', False)

	dict_1 = torch.load('model_1.pt')
	dict_2 = torch.load('model_2.pt')
	cnn_1.load_state_dict(dict_1)
	cnn_2.load_state_dict(dict_2)
	trained_1_list = list(dict_1.keys())
	trained_2_list = list(dict_2.keys())
	j = 0
	d_pos = []
	for i in range(5):
		while str.find(trained_1_list[j], 'conv{}'.format(i+1)) != 0:
			j += 1
		d_pos.append(j)

	for l in range(1,5):
		for r in range(l + 1, 5):

			cnn = CNN()
			dict_prime = cnn.state_dict().copy()
			new_list = list(cnn.state_dict().keys())
			for i in range(d_pos[l]):
				dict_prime[new_list[i]] = dict_1[trained_1_list[i]]
			for i in range(d_pos[r], len(dict_prime)):
				dict_prime[new_list[i]] = dict_2[trained_2_list[i]]
			cnn.load_state_dict(dict_prime)

			for name, param in cnn.named_parameters():
				for i in list(range(l)) + list(range(r, 5)):
					if 'conv%d' % (i + 1) in name:
						param.requires_grad = False
				if 'out' in name:
					param.requires_grad = False

			calc(cnn, 1, 'A->B_{}_{}_1'.format(l, r), True, cnn_2, r - 1)
			#calc(cnn, 0, 'A->B_{}_{}_0'.format(l, r), True, cnn_2, r - 1)
			#calc(cnn, 0.0007, 'A->B_{}_{}_mr'.format(l, r), True, cnn_2, r - 1)

			cnn = CNN()
			dict_prime = cnn.state_dict().copy()
			new_list = list(cnn.state_dict().keys())
			for i in range(d_pos[l]):
				dict_prime[new_list[i]] = dict_2[trained_2_list[i]]
			for i in range(d_pos[r], len(dict_prime)):
				dict_prime[new_list[i]] = dict_1[trained_1_list[i]]
			cnn.load_state_dict(dict_prime)

			for name, param in cnn.named_parameters():
				for i in list(range(l)) + list(range(r, 5)):
					if 'conv%d' % (i + 1) in name:
						param.requires_grad = False
				if 'out' in name:
					param.requires_grad = False

			calc(cnn, 1, 'B->A_{}_{}_1'.format(l, r), True, cnn_1, r - 1)
			#calc(cnn, 0, 'B->A_{}_{}_0'.format(l, r), True, cnn_1, r - 1)
			#calc(cnn, 0.07, 'B->A_{}_{}_mr'.format(l, r), True, cnn_1, r - 1)
