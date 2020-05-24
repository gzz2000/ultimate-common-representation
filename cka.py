import torch
from cifar10 import CNN, test_data, device
import numpy as np

def feature_space_linear_cka(X, Y):
	X -= np.mean(X, 0, keepdims=True)
	Y -= np.mean(Y, 0, keepdims=True)
	dot_sim = np.linalg.norm(X.T.dot(Y)) ** 2
	norm_x = np.linalg.norm(X.T.dot(X))
	norm_y = np.linalg.norm(Y.T.dot(Y))
	return dot_sim / (norm_x * norm_y)

if __name__ == '__main__':
	test_loader = torch.utils.data.DataLoader(test_data, 2000)

	feat1 = []
	feat2 = []

	def hook1(module, input, output):
		feat1.append(output.view(output.size(0), -1).clone().detach().cpu().numpy())

	def hook2(module, input, output):
		feat2.append(output.view(output.size(0), -1).clone().detach().cpu().numpy())


	cnn1 = CNN()
	cnn1.to(device)
	cnn1.eval()
	cnn2 = CNN()
	cnn2.to(device)
	cnn2.eval()

	cnn1.load_state_dict(torch.load('model_1.pt'))
	cnn2.load_state_dict(torch.load('model3.pt'))

	for layer in cnn1.conv_layers:
		layer.register_forward_hook(hook1)
	for layer in cnn2.conv_layers:
		layer.register_forward_hook(hook2)

	x, y = next(iter(test_loader))
	x = x.to(device)
	with torch.no_grad():
		cnn1(x)
		cnn2(x)
	for X in feat1:
		for Y in feat2:
			print(feature_space_linear_cka(X, Y), end=' ')
		print("")
