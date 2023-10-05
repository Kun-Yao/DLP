import pandas as pd
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import skimage
from skimage import io, exposure
from skimage.transform import resize
import os

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv', header=None)
        label = pd.read_csv('train_label.csv', header=None)
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv', header=None)
        label = pd.read_csv('test_label.csv', header=None)
        return np.squeeze(img.values), np.squeeze(label.values)

def save():
    for i in range(len(train_data)):
        io.imsave(f'C:/Users/Lab639/Desktop/deep learning/dllab4/cut_train/{train_data.img_name[i]}.jpeg', train_data[i][0])
    #img = loader[i][0]
    for i in range(len(test_data)):
        io.imsave(f'C:/Users/Lab639/Desktop/deep learning/dllab4/cut_test/{test_data.img_name[i]}.jpeg', test_data[i][0])


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        image_path= f'{self.root}/{self.img_name[index]}.jpeg'
        #print(image_path)
        label = self.label[index]
        img = io.imread(image_path) #(H, W, C)

        
        """
        Data preprocessing
        1.提高影像對比度(找眼球邊界)
        2.切下眼球範圍
        3.補齊正方形
        4.resize 512*512*3
        """
        gamma = exposure.adjust_gamma(img, gamma=0.7)
        
        col_mean = np.mean(gamma, axis=0)
        row_mean = np.mean(gamma, axis=1)
        thre = 30
        left = np.where(col_mean > thre)[0][0]
        right = np.where(col_mean > thre)[0][-1]
        up = np.where(row_mean > thre)[0][0]
        down = np.where(row_mean > thre)[0][-1]
        cut = img[up:down+1, left:right+1, :]
        v_dist = down-up+1
        h_dist = right-left+1
        if v_dist < h_dist:
            padding = np.zeros((((h_dist-v_dist) / 2).astype(np.int32), h_dist, 3))
            cut = np.concatenate((padding, cut), axis=0)
            cut = np.concatenate((cut, padding))
        cut = np.resize(cut, (512, 512, 3))
        return gamma, label


train_data = RetinopathyLoader('C:/Users/Lab639/Desktop/deep learning/dllab4/new_train', 'train')
test_data = RetinopathyLoader('C:/Users/Lab639/Desktop/deep learning/dllab4/new_test', 'test')
save()
