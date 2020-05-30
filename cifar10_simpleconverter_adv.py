import torch
import torch.nn as nn
from tqdm import tqdm
from cifar10 import CNN
from cifar10_simpleconverter import SimpleConv, Ensemble
from data_cifar10 import *
from test_acc import print_acc

if __name__ == '__main__':
    device = torch.device('cuda')

    model1 = CNN().to(device)
    model2 = CNN().to(device)
    
    INDEX1 = 1
    INDEX2 = 3

    conv12 = SimpleConv(INDEX1, INDEX2).to(device)
    conv21 = SimpleConv(INDEX1, INDEX2).to(device)

    ens12 = Ensemble(model1, model2, INDEX1, INDEX2, conv12)
    ens21 = Ensemble(model2, model1, INDEX1, INDEX2, conv21)

    EPOCH = 50
    optim_conv = torch.optim.Adam(list(conv12.parameters()) + list(conv21.parameters()))
    loss_f_conv_l1 = nn.L1Loss()
    loss_f_conv_l2 = nn.MSELoss()
    optim_cl = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()))
    loss_f_cl = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCH):
        run_loss_models = 0
        run_loss_convs = 0
        
        model1.train()
        model2.train()
        conv12.train()
        conv21.train()
        for x, y in tqdm(train_loader, desc = 'Epoch {:d}/{:d}'.format(epoch, EPOCH)):
            x = x.to(device)
            y = y.to(device)

            # train classifier
            optim_cl.zero_grad()
            ypred1 = model1(x)
            ypred2 = model2(x)
            loss_cl = loss_f_cl(ypred1, y) + loss_f_cl(ypred2, y)
            loss_cl.backward(retain_graph = True)

            f11 = model1.feat[INDEX1 - 1]
            f12 = model1.feat[INDEX2 - 1]
            f21 = model2.feat[INDEX1 - 1]
            f22 = model2.feat[INDEX2 - 1]
            pred_f22 = conv12(f11)
            pred_f12 = conv21(f21)
            loss_conv = torch.sqrt(loss_f_conv_l1(pred_f22, f22)) + torch.sqrt(loss_f_conv_l1(pred_f12, f12))
            loss_conv = (-1) * loss_conv
            loss_conv.backward()
            optim_cl.step()

            run_loss_models += loss_cl.item()

            # train converter
            optim_conv.zero_grad()
            model1(x)
            model2(x)
            f11 = model1.feat[INDEX1 - 1].detach()
            f12 = model1.feat[INDEX2 - 1].detach()
            f21 = model2.feat[INDEX1 - 1].detach()
            f22 = model2.feat[INDEX2 - 1].detach()
            pred_f22 = conv12(f11)
            pred_f12 = conv21(f21)
            loss_conv = loss_f_conv_l2(pred_f22, f22) + loss_f_conv_l2(pred_f12, f12)
            loss_conv.backward()
            optim_conv.step()

            run_loss_convs += loss_conv.item()

        print('Train Loss of Models: {:.5f}'.format(run_loss_models / len(train_loader)))
        print('Train Loss of Converters: {:.5f}'.format(run_loss_convs / len(train_loader)))

        if run_loss_convs / len(train_loader) > 10:
            # re-train converter?
            print('CONVERTER RESET')
            conv12 = SimpleConv(INDEX1, INDEX2).to(device)
            conv21 = SimpleConv(INDEX1, INDEX2).to(device)
            optim_conv = torch.optim.Adam(list(conv12.parameters()) + list(conv21.parameters()))

            EPOCHRST = 3

            for epochrst in range(EPOCHRST):
                run_loss_convs = 0
                for x, y in tqdm(train_loader, desc = 'EpochRst {:d}/{:d}'.format(epochrst, EPOCHRST)):
                    x = x.to(device)
                    y = y.to(device)

                    # train converter
                    optim_conv.zero_grad()
                    model1(x)
                    model2(x)
                    f11 = model1.feat[INDEX1 - 1].detach()
                    f12 = model1.feat[INDEX2 - 1].detach()
                    f21 = model2.feat[INDEX1 - 1].detach()
                    f22 = model2.feat[INDEX2 - 1].detach()
                    pred_f22 = conv12(f11)
                    pred_f12 = conv21(f21)
                    loss_conv = loss_f_conv_l2(pred_f22, f22) + loss_f_conv_l2(pred_f12, f12)
                    loss_conv.backward()
                    optim_conv.step()

                    run_loss_convs += loss_conv.item()

                print('[RESET]Train Loss of Converters: {:.5f}'.format(run_loss_convs / len(train_loader)))

        print_acc(model1, 'model1')
        print_acc(model2, 'model2')
        print_acc(ens12, 'ens12')
        print_acc(ens21, 'ens21')

        if (epoch + 1) % 5 == 0:
            torch.save(model1.state_dict(), './models/CIFAR10_no_dropout/advsc/model_3e_epoch_{:d}_1.pt'.format(epoch))
            torch.save(model2.state_dict(), './models/CIFAR10_no_dropout/advsc/model_3e_epoch_{:d}_2.pt'.format(epoch))
