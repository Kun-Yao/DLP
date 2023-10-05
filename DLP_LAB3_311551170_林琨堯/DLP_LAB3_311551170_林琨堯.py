# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:39:01 2023

@author: Lin
"""


import dataloader as dl
import torch
from torch.utils import data
from torch import optim
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class EEGNet(nn.Module):
    def __init__(self, activation = nn.ELU()):
        super(EEGNet, self).__init__()
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1,
                      out_channels = 16,
                      kernel_size = (1, 51),
                      stride = (1, 1),
                      padding = (0, 25),
                      bias = False),
            nn.BatchNorm2d(16,
                           eps = 1e-05,
                           momentum = 0.1,
                           affine = True,
                           track_running_stats = True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels = 16,
                      out_channels = 32,
                      kernel_size = (2, 1),
                      stride = (1, 1),
                      groups = 16,
                      bias = False),
            nn.BatchNorm2d(32,
                           eps = 1e-05,
                           momentum = 0.1,
                           affine = True,
                           track_running_stats = True),
            #nn.ELU(alpha = 1.0),
            activation,
            nn.AvgPool2d(kernel_size = (1, 4),
                         stride = (1, 4),
                         padding = 0),
            nn.Dropout(p = 0.25)
            )
        
        self.separable_conv = nn.Sequential(
            nn.Conv2d(in_channels = 32,
                      out_channels = 32,
                      kernel_size = (1, 15),
                      stride = (1, 1),
                      padding = (0, 7),
                      bias = False),
            nn.BatchNorm2d(32,
                           eps = 1e-05,
                           momentum = 0.1,
                           affine = True,
                           track_running_stats = True),
            #nn.ELU(alpha = 1.0),
            activation,
            nn.AvgPool2d(kernel_size = (1, 8),
                         stride = (1, 8),
                         padding = 0),
            nn.Dropout(p=0.25)
            )
        
        self.classify = nn.Sequential(
            nn.Linear(in_features = 736, 
                      out_features = 2,
                      bias = True)
            )
          
    def forward(self, input):
        out = self.first_conv(input)
        out = self.depthwise_conv(out)
        out = self.separable_conv(out)
        # print(f'separable_conv {out.shape}')
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        # print(out.shape)
        return out
    
    
class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        #boader: dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (2, 1)),
            nn.BatchNorm2d(25, eps = 1e-05, momentum = 0.1),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
            )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50, eps = 1e-05, momentum = 0.1),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
            )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100 ,eps = 1e-05, momentum = 0.1),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
            )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200, eps = 1e-05, momentum = 0.1),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
            )
        
        self.classify = nn.Linear(8600, 2)
    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        # print(out.shape)
        return out

def plot_curve(train, test, epoch):
    plt.plot(epoch, train, label = '')
    plt.plot(epoch, test)
    
#%%
def Train_EEG(activation, epoch, lr, device):
    Loss = nn.CrossEntropyLoss()
    
    for key, value in activation.items():
        print(key)
        net = EEGNet(value)
        net.to(device)
        train_acc = []
        train_loss = []
        test_acc = []
        Epoch = []
        
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
        
        for i in range(1, epoch+1):
            total_train_accuracy = 0
            total_train_loss = 0
            net.train()
            for idx,(datas,label) in enumerate(loader_train):
                datas=datas.to(device,dtype=torch.float)
                label=label.to(device,dtype=torch.long)
                pred = net(datas)
                
                loss = Loss(pred, label)
                
                total_train_loss = total_train_loss + loss.item()
                accuracy = (pred.argmax(1) == label).sum()
                total_train_accuracy = total_train_accuracy + accuracy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_acc.append((total_train_accuracy / len(loader_train.dataset)).cpu().data.numpy())
            train_loss.append(total_train_loss / len(loader_train.dataset))
            
            if i%100==0:
                    print(f'epcoh{i}  train_loss:{train_loss[-1]}  train_acc:{train_acc[-1]}')
            
            
            net.eval()
            with torch.no_grad():
                total_test_accuracy = 0
                for idx,(datas,target) in enumerate(loader_test):
                    datas=datas.to(device,dtype=torch.float)
                    label=label.to(device,dtype=torch.long)
                    pred = net(datas)
                    
                    accuracy = (pred.argmax(1) == label).sum()
                    total_test_accuracy += accuracy
                    
                test_acc.append((total_test_accuracy/len(loader_test.dataset)).cpu().data.numpy())
                
                if test_acc[i-1] > highest_acc['eeg'][key]:
                    highest_acc['eeg'][key] = test_acc[i-1]
                    
                if i%100==0:
                        print(f'test_acc:{test_acc[-1]}')
                
                if test_acc[-1] > 0.9:
                    break
            
            Epoch.append(i)
        plt.title('Activation function Comparision (EEG)')
        plt.plot(epoch, train_acc, label = f'{key}_train')
        plt.plot(epoch, test_acc, label = f'{key}_test')
    plt.show()
#%%
def Train_DCN(activation, epoch, lr, device):
    Loss = nn.CrossEntropyLoss()
    
    for key, value in activation.items():
        print(key)
        net = DeepConvNet(value)
        net.to(device)
        train_acc = []
        train_loss = []
        test_acc = []
        Epoch = []
        
        optimizer = optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
        
        for i in range(1, epoch+1):
            total_train_accuracy = 0
            total_train_loss = 0
            net.train()
            for idx,(datas,label) in enumerate(loader_train, 0):
                datas=datas.to(device,dtype=torch.float)
                label=label.to(device,dtype=torch.long) 
                pred = net(datas)
                
                loss = Loss(pred, label)
                
                total_train_loss = total_train_loss + loss.item()
                accuracy = (pred.argmax(1) == label).sum()
                total_train_accuracy = total_train_accuracy + accuracy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_acc.append((total_train_accuracy / len(loader_train.dataset)).cpu().data.numpy())
            train_loss.append(total_train_loss / len(loader_train.dataset))
            
            if i%100==0:
                    print(f'epcoh{i}  train_loss:{train_loss[-1]}  train_acc:{train_acc[-1]}')
            
            
            net.eval()
            with torch.no_grad():
                total_test_accuracy = 0
                for idx,(datas,target) in enumerate(loader_test, 0):
                    datas=datas.to(device,dtype=torch.float)
                    label=label.to(device,dtype=torch.long)
                    pred = net(datas)
                    
                    accuracy = (pred.argmax(1) == label).sum()
                    total_test_accuracy += accuracy
                    
                test_acc.append((total_test_accuracy/len(loader_test.dataset)).cpu().data.numpy())
                
                if test_acc[i-1] > highest_acc['deepconv'][key]:
                    highest_acc['deepconv'][key] = test_acc[i-1]
                    
                if i%100==0:
                        print(f'epcoh{i}  test_acc:{test_acc[-1]}')
                
                if test_acc[-1] > 0.9:
                    break
            
            Epoch.append(i)
        plt.title('Activation function Comparision (Deep)')
        plt.plot(epoch, train_acc, label = f'{key}_train')
        plt.plot(epoch, test_acc, label = f'{key}_test')
    plt.show()
#%%            
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

train_data, train_label, test_data, test_label = dl.read_bci_data()
train=TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
# train = np.concatenate(train_data, train_label)
loader_train=DataLoader(train,batch_size=30,shuffle=True,num_workers=4)
test=TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
# test = np.concatenate(test_data, test_label)
loader_test=DataLoader(test,batch_size=30,shuffle=False,num_workers=4)
# print(f'test dataset:\n{loader_test}')

models = ['eeg', 'deepconv']
acts = {'relu':nn.ReLU(), 'leaky_relu':nn.LeakyReLU(), 'elu':nn.ELU()}

# train = list(zip(train_data, train_label))
# test = list(zip(test_data, test_label))

highest_acc = {'eeg':{'relu':0, 'leaky_relu':0, 'elu':0}, 
               'deepconv':{'relu':0, 'leaky_relu':0, 'elu':0}}

if __name__ == '__main__':
    epoch = 100
    lr = 0.01
    
    # Train(models[1], 'elu', acts['elu'], epoch, lr)
    Train_EEG(acts, epoch, lr, device)
    
    Train_DCN(acts, epoch, lr, device)
    # for m in models:
    #     for a, v in acts.items():
    #         Train(m, a, v, epoch, lr)
    #         plot_curve(train_acc, test_acc)
    # plt.show()