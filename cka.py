import torch
from cifar10 import CNN, test_data, device
import numpy as np

test_loader = torch.utils.data.DataLoader(test_data, 2000)

def feature_space_linear_cka(X, Y):
	X = X - torch.mean(X, 0, keepdims=True)
	Y = Y - torch.mean(Y, 0, keepdims=True)
	dot_sim = X.T.matmul(Y).norm() ** 2
	norm_x = X.T.matmul(X).norm()
	norm_y = Y.T.matmul(Y).norm()
	return dot_sim / (norm_x * norm_y)

def each_layer_pair_cka(cnn1, cnn2):
	cnn1.eval()
	cnn2.eval()
	x, y = next(iter(test_loader))
	x = x.to(device)
	with torch.no_grad():
		cnn1(x)
		cnn2(x)
	ret = np.array([[feature_space_linear_cka(X, Y).item() for Y in cnn2.feat] for X in cnn1.feat])
	torch.cuda.empty_cache()
	return ret

if __name__ == '__main__':

	cnn1 = CNN()
	cnn1.to(device)
	cnn1.eval()
	cnn2 = CNN()
	cnn2.to(device)
	cnn2.eval()

	cnn1.load_state_dict(torch.load('model_1.pt'))
	cnn2.load_state_dict(torch.load('model_2.pt'))

	print(each_layer_pair_cka(cnn1, cnn2))
