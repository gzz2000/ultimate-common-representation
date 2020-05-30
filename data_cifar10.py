import torch.utils.data as Data
import torchvision

BATCH_SIZE = 64
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=10)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=10)

