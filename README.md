# ultimate-common-representation

ML course project: investigation on common perceptions of same neural network model with different random seed

## Install PyTorch
https://pytorch.org/get-started/locally/

Caution: If you use Ubuntu 16.04 with Python 3.5.2, torchvision 0.6.0 and torch 1.5.0 [won't work](https://github.com/pytorch/vision/issues/2132). Please consider downgrading.

## Use
For MNIST experiments, run jupyter notebook and open `basic.ipynb`.
For CIFAR10 experiments, run `cifar10.py` to train one model, and run `cifar10_converter_adv.py` to train a converter (please see the code for details).
