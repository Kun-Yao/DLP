# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:07:41 2023

@author: Lab639
"""

import torch
import numpy as np
from dataloader import RetinopathyLoader
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from dataloader import RetinopathyLoader
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaulate(model, test_loader, device, num_class):
    confusion_matrix=np.zeros((num_class,num_class))
    with torch.set_grad_enabled(False):
        model.eval()
        correct = 0
        for _, (images, label) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            predict = model(images)
            pred = predict.argmax(dim=1)
            for i in range(len(label)):
                confusion_matrix[int(label[i])][int(pred[i])]+=1
                if pred[i] == label[i]:
                    correct +=1
        acc = 100. * correct / len(test_loader.dataset)
    #normalize
    confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_class,1)
    return acc, confusion_matrix


batch_size_50 = 8
if __name__== "__main__":
    print(torch.cuda.is_available())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset=RetinopathyLoader('gamma_test',mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size_50, shuffle=False, num_workers=4)

    # model=models.resnet18(pretrained=True)
    # model.load_state_dict(torch.load(os.path.join('./models','resnet18_with_pretrain.pth')))

    model=models.resnet50(pretrained=True)
    model.load_state_dict(torch.load(os.path.join('resnet50_with_pretrain.pth')))


    model=model.to(device)
    acc,_ = evaulate(model, test_loader, device, num_class=5)
    # print("resnet18_with_pretrain.pth:", acc, "%")
    print("resnet50_with_pretrain.pth:", acc, "%")