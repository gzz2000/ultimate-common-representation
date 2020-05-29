import torch
import math
import pynvml
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from tqdm import trange

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
pynvml.nvmlInit()
torch.manual_seed(721)
device = torch.device('cuda')

EPOCH = 20
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.CIFAR10(
    root='../data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.CIFAR10(
    root='../data', train=False, transform=torchvision.transforms.ToTensor())
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
        self.layers = [self.conv1, self.conv2,
                       self.conv3, self.conv4, self.conv5, self.out]
        self.bid = 5

    def forward(self, x):
        self.feat.clear()
        for layer in range(self.bid):
            x = self.layers[layer](x)
            self.feat.append(x.view(x.size(0), -1))
        if(self.bid == 5):
            output = self.layers[5](x).view(x.size(0), 10)
            self.feat.append(output.view(output.size(0), -1))
            return output


# np.random.seed(721234827)
C = 0.57721566490153286060651209008240243104215933593992
pi = 3.14159265358979323846264338327950288419716939937510
dim = [32768, 8192, 24576, 6144, 6144, 10]
S_limit = []
for i in range(6):
    S_limit.append((72000000/6)/dim[i])
S_limit[5] = 5999
max_norm = np.zeros([6])
# print(S_limit)
X = []
NER_K = 3
Cnt = np.zeros([6, 10])
for i in range(6):
    X.append([])
    for j in range(10):
        X[i].append([])
# cnn_1.to(device)
# x,y=next(iter(train_loader))
# x=x.to(device)
# with torch.no_grad():
#    cnn_1(x)
# S_n=np.zeros([10],dtype=int)
# X=[]
# dim=[]
# re_dim=[]
# S_de=np.zeros([6],dtype=bool)
# D_limit=5000
# S_limit=700
# MAX_limit=32768
# vec=np.zeros([D_limit,MAX_limit])
# for i in range(D_limit):
#    vec[i]=np.random.normal(0,1,MAX_limit)
# for i in range(6):
#    tmp=cnn_1.feat[i].cpu().numpy().shape[1]
#    re_dim.append(tmp)
#    X.append([])
#    if(tmp>D_limit):
#        S_de[i]=True
#        dim.append(D_limit)
#        for j in range(10):
#            X[i].append(np.zeros([S_limit,D_limit]))
#    else:
#        dim.append(tmp)
#        for j in range(10):
#            X[i].append(np.zeros([S_limit,tmp]))


def entropy(X, max_norm):
    n, m = X.shape
    X = torch.mm(X, X.T)
    D = torch.diag(X).repeat(n, 1)
    D = D+D.T-X*2
    L_sum = 0
    for k in range(1, NER_K):
        L_sum -= (NER_K-k)/k
    D = D.sqrt().log()
    D = D.sort(dim=1)[0]
    if m > 27000:
        m = math.floor(math.log(m)/(0.03**2/2-0.03**3/3))
    else:
        if m > 10000:
            m = math.floor(math.log(m)/(0.05**2/2-0.05**3/2))
    if m != 10:
        H = n*NER_K*math.log(math.sqrt(m)/max_norm)
    else:
        H = 0
    for i in range(n):
        H += D[i][1:NER_K+1].sum().item()
    if ((m % 2) == 0):
        tmp_gamma = m/2.0*math.log(pi)
        for i in range(m//2):
            tmp_gamma -= math.log(i+1)
    else:
        tmp_gamma = (m-1)/2.0*math.log(pi)
        for i in range(1+m//2):
            tmp_gamma -= math.log(i+0.5)
    return m*H/(n*NER_K)+C+math.log(n)+tmp_gamma-L_sum/NER_K


def calc(cnn, mu, Str, fg,  pro=None, pos=None):
    if(fg == True):
        cnn.bid = pos+1
        pro.bid = pos+1
        pro.to(device)
        pro.train()
    else:
        cnn.to(device)
    cnn.train()

    optimizer = torch.optim.Adam(
        filter(lambda q: q.requires_grad, cnn.parameters()), lr=LR)

    loss_f = nn.CrossEntropyLoss()

    if(fg == True):
        EPOCH = 1
    else:
        EPOCH = 20
    iteration = trange(EPOCH)
    for epoch in iteration:
        #        running_correct = 0
        running_loss = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
#            outputs = cnn(b_x)
            if(fg == True):
                cnn(b_x)
                with torch.no_grad():
                    pro(b_x)
#                loss = (1-mu)*loss_f(outputs, b_y)+mu * torch.dist(cnn.feat[pos], pro.feat[pos], p=2)
                loss = torch.dist(cnn.feat[pos], pro.feat[pos], p=2)
            else:
                outputs = cnn(b_x)
                loss = loss_f(outputs, b_y)

            optimizer.zero_grad()
#            y_pred = torch.max(outputs, 1).indices

            loss.backward()
            running_loss += loss.item()
#            running_correct += torch.sum(y_pred == b_y).item()
            optimizer.step()
        iteration.set_description(str(running_loss / len(train_loader)))
        torch.cuda.empty_cache()

    if(fg == False):
        torch.save(cnn.state_dict(), 'model_{}.pt'.format(Str))
    return est(cnn, mu, Str, fg, pro, pos)


def est(cnn, mu, Str, fg,  pro=None, pos=None, H=None):
    global max_norm
    if(fg == True):
        cnn.bid = 5
        pro.bid = pos+1
        pro.to(device)
        pro.eval()
    else:
        cnn.to(device)
    loss_f = nn.CrossEntropyLoss()
    cnn.eval()
    test_correct = 0
    test_loss = 0
    train_correct = 0
    train_loss = 0
    i = 1
    tr = 0
    with torch.no_grad():

        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            outputs = cnn(X_train)
            if(fg == True):
                pro(X_train)
#                loss = (1-mu)*loss_f(outputs, y_train)+mu *
                loss = torch.dist(cnn.feat[pos], pro.feat[pos], p=2)
                tr += 1-loss/pro.feat[pos].norm(p=2)
            else:
                loss = loss_f(outputs, y_train)
                label = y_train.cpu().numpy()
#                for i in range(6):
#                    tmp.append(cnn.feat[i].cpu().detach().numpy())
#                for j in range(len(y_train)):
#                    y=label[j]
#                    S_n[y]+=1
#                    if(S_n[y]<=S_limit):
#                        for i in range(6):
#                            if(S_de[i]==True):
#                                for k in range(dim[i]):
#                                    X[i][y][S_n[y]][k]=np.dot(tmp[i][j],vec[k][0:re_dim[i]].T)
#                            else:
#                                X[i][y][S_n[y]]=tmp[i][j]
                for j in range(label.size):
                    y = label[j]
                    for i in range(6):
                        X[i][y].append(cnn.feat[i][j])
                        if max_norm[i] < cnn.feat[i][j].norm(p=2):
                            max_norm[i] = cnn.feat[i][j].norm(p=2)
                        tmp = len(X[i][y])
                        if(tmp > S_limit[i]):
                            Q = torch.full(
                                [tmp, X[i][y][0].size()[0]], 0, dtype=torch.float32).to(device)
                            for k in range(tmp):
                                Q[k] = X[i][y][k]
#                            H_tmp = entropy(Q)
                            Cnt[i][y] += 1
#                            print(i)
#                            print(y)
#                            print(H_tmp)
#                            print('\n')
                            H[i][y] += entropy(Q, max_norm[i])
                            X[i][y] = []

            y_pred = torch.max(outputs, 1).indices

            train_correct += torch.sum(y_pred == y_train).item()
            train_loss += loss.item()

        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            outputs = cnn(X_test)
            if(fg == True):
                pro(X_test)
#                loss = (1-mu)*loss_f(outputs, y_test)+mu *
                loss = torch.dist(cnn.feat[pos], pro.feat[pos], p=2)
                tr += 1-loss/pro.feat[pos].norm(p=2)
            else:
                loss = loss_f(outputs, y_test)
                label = y_test.cpu().numpy()
                for j in range(label.size):
                    y = label[j]
                    for i in range(6):
                        X[i][y].append(cnn.feat[i][j])
                        if max_norm[i] < cnn.feat[i][j].norm(p=2):
                            max_norm[i] = cnn.feat[i][j].norm(p=2)
                        tmp = len(X[i][y])
                        if(tmp > S_limit[i]):
                            Q = torch.full(
                                [tmp, X[i][y][0].size()[0]], 0, dtype=torch.float32).to(device)
                            for k in range(tmp):
                                Q[k] = X[i][y][k]
#                            H_tmp = entropy(Q)
                            Cnt[i][y] += 1
#                            print(i)
#                            print(y)
#                            print(H_tmp)
#                            print('\n')
                            H[i][y] += entropy(Q, max_norm[i])
                            X[i][y] = []

            y_pred = torch.max(outputs, 1).indices

            test_correct += torch.sum(y_pred == y_test).item()
            test_loss += loss.item()

    torch.cuda.empty_cache()

    if (fg == False):
        for i in range(6):
            for y in range(10):
                tmp = len(X[i][y])
                if(tmp > 0.5*S_limit[i]):
                    Q = torch.full([tmp, X[i][y][0].size()[0]],
                                   0, dtype=torch.float32).to(device)
                    for k in range(tmp):
                        Q[k] = X[i][y][k]
                    Cnt[i][y] += 1
                    H[i][y] += entropy(Q, max_norm[i])
                    X[i][y] = []
                H[i][y] /= Cnt[i][y]
    print(Str)
    print('Train Loss {:.4f}'.format(train_loss / len(train_loader)))
    print('Test Loss {:.4f}'.format(test_loss / len(test_loader)))

    print('Train Acc {:.4f}%'.format(train_correct / len(train_data) * 100))
    print('Test Acc {:.4f}%'.format(test_correct / len(test_data) * 100))

    if(fg == True):
        #        print(tr)
        return tr/(len(train_loader)+len(test_loader))


if __name__ == '__main__':

    cnn_1 = CNN()
    cnn_2 = CNN()
    H_1 = np.zeros([6, 10])
    H_2 = np.zeros([6, 10])

    #    calc(cnn_1, 0, '1', False)
    #    calc(cnn_2, 0, '2', False)

    cnn_1.load_state_dict(torch.load('model_1.pt', map_location='cpu'))
    cnn_2.load_state_dict(torch.load('model_2.pt', map_location='cpu'))

#    est(cnn_1, 0, '1', False, H=H_1)
#    est(cnn_2, 0, '2', False, H=H_2)
#    np.savez('entropy.npz', H_1=H_1, H_2=H_2)

    H = np.load('entropy.npz')
    H_1 = H['H_1']
    H_2 = H['H_2']
#    for i in range(6):
#        print(H_1[i])
#        print(H_2[i])

    for name, param in cnn_1.named_parameters():
        param.requires_grad = False
    for name, param in cnn_2.named_parameters():
        param.requires_grad = False
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(meminfo.used)

#    convert_ratio = np.zeros([6, 6])
    convert_ratio = np.load('convert_ratio.npy')
    print(convert_ratio)

    for l in range(5, 5):
        for r in range(l+1, 6):
            # mask l+1..r, l-->r
            cnn = CNN()
            for k in range(l+1):
                cnn.layers[k] = cnn_1.layers[k]
                cnn_1.layers[k].to(device)
            for k in range(l+1, r+1):
                cnn_1.layers[k].to(torch.device('cpu'))
                cnn.layers[k].to(device)
            for k in range(r+1, 6):
                cnn.layers[k] = cnn_2.layers[k]
                cnn_1.layers[k].to(torch.device('cpu'))

            convert_ratio[l][r] = calc(cnn, 1, 'A->B_{}_{}_1'.format(l, r),
                                       True, cnn_2, r)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(meminfo.used)

            cnn = CNN()
            for k in range(l+1):
                cnn.layers[k] = cnn_2.layers[k]
                cnn_2.layers[k].to(device)
            for k in range(l+1, r+1):
                cnn_2.layers[k].to(torch.device('cpu'))
                cnn.layers[k].to(device)
            for k in range(r+1, 6):
                cnn.layers[k] = cnn_1.layers[k]
                cnn_2.layers[k].to(torch.device('cpu'))

            convert_ratio[r][l] = calc(cnn, 1, 'B->A_{}_{}_1'.format(l, r),
                                       True, cnn_1, r)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(meminfo.used)

            np.save('convert_ratio.npy', convert_ratio)
            print('loss_saved_%d' % (l))
#        exit(0)

# similarity
# to be modified
    # print(convert_ratio)
    # print(H_1)
    # print(H_2)
    L6_same = 0
    cnn_1.to(device)
    cnn_2.to(device)
    with torch.no_grad():

        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            outputs_1 = cnn_1(X_train)
            outputs_2 = cnn_2(X_train)
            pred_1 = torch.max(outputs_1, 1).indices
            pred_2 = torch.max(outputs_2, 1).indices

            L6_same += torch.sum(pred_1==pred_2).item()

        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            outputs_1 = cnn_1(X_test)
            outputs_2 = cnn_2(X_test)
            pred_1 = torch.max(outputs_1, 1).indices
            pred_2 = torch.max(outputs_2, 1).indices
            
            L6_same += torch.sum(pred_1==pred_2).item()
    L6_same/=len(train_data)+len(test_data)

    si = np.zeros([6, 6])
    for l in range(6):
        for r in range(6):
            if (l > r):
                H_r = 0
                for k in range(10):
                    H_r += H_1[l][k]/H_2[r][k]
                H_r /= 10
                si[l][r] = min(1, convert_ratio[l][r]*min(H_r, 1))
            else:
                if(l < r):
                    H_r = 0
                    for k in range(10):
                        H_r += H_2[r][k]/H_1[l][k]
                    H_r /= 10
                    si[l][r] = min(1, convert_ratio[l][r]*min(H_r, 1))
                else:
                    if(l == 5 and r == 5):
                        H_r = 0
                        for k in range(10):
                            H_r += H_1[l][k]/H_2[r][k]
                        H_r /= 10
                        si[l][r] = min(H_r, 1/H_r)*L6_same
                    else:
                        H_AB = 0
                        for k in range(10):
                            H_AB += H_1[l][k]/H_2[r][k]
                        H_AB /= 10
                        tmp_sim = 0
                        for k in range(5-l):
                            tmp_sim+=max(convert_ratio[l][r+k+1]*min(H_AB, 1), convert_ratio[l+k+1][r]*min(1, 1/H_AB))
                        si[l][r] = min(1, tmp_sim/(5-l))
    print(si)
    np.savetxt('convert_ratio.txt', convert_ratio, fmt='%0.17f', delimiter=' ')
    np.savetxt('H_1.txt', H_1, fmt='%17.7f', delimiter=' ')
    np.savetxt('H_2.txt', H_2, fmt='%17.7f', delimiter=' ')
    np.savetxt('similarity.txt', si, fmt='%.17f', delimiter=' ')
