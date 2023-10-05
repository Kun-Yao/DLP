import json
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
from PIL import Image
from sklearn import preprocessing
import os
import numpy as np

train_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
class CustomDataset(Dataset):
    def __init__(self, data_file, mode):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        with open('objects.json', 'r') as f:
            self.label_index = json.load(f)
        self.transform = train_tfm
        self.mode = mode
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if(self.mode == 'train'):             
            image_path = list(self.data.keys())[idx]
            labels = self.data[image_path]
            path = "iclevr/iclevr/" +  image_path
            self.img = cv2.imread(path)
            self.img = self.transform(self.img)
        else:
            self.img = torch.ones(1)
            labels =list(self.data)[idx]
        
        label_indices = [self.label_index[label] for label in labels]
        one_hot_vector = torch.zeros(len(self.label_index))
        one_hot_vector[label_indices] = 1
        
        return self.img, one_hot_vector

def getCode(root):
    path = os.path.join(root, 'objects.json')
    with open(path) as file:
        code = json.load(file)
    return code

def getTrainData(root, mode, code):
    path = os.path.join(root, mode+'.json')
    with open(path) as file:
        data = json.load(file)

    lb = preprocessing.LabelBinarizer()
    lb.fit([i for i in range(24)])

    # make data
    img_name = []
    labels = []
    for key, value in data.items():
        img_name.append(key)
        tmp = []
        for i in range(len(value)):
            tmp.append(np.array(lb.transform([code[value[i]]])))
        labels.append((np.sum(tmp, axis=0)))
    print("train_img_name:", len(img_name))
    print("train_labels:", len(labels))
    labels = torch.tensor(np.array(labels))
    #print(labels[0])
    return img_name, labels

def getTestData(root, mode, code):
    path = os.path.join(root, mode+'.json')
    with open(path) as file:
        data = json.load(file)
    lb = preprocessing.LabelBinarizer()
    lb.fit([i for i in range(24)])
    # make data
    labels = []
    for value in data:
        tmp = []
        for i in range(len(value)):
            tmp.append(np.array(lb.transform([code[value[i]]])))
        labels.append(np.sum(tmp, axis=0))
    print("test_labels:", len(labels))
    labels =torch.tensor(np.array(labels))
    #print(labels[0])
    return labels

class iclevrLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.code = getCode(root)
        # get data
        if mode=="train":
            self.img_name, self.label = getTrainData(root, mode, self.code)
        elif mode=="test" or mode=="new_test":
            self.label = getTestData(root, mode, self.code)
        else:
            raise ValueError("No such root!")
        self.mode = mode
        #self.check()

    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)

    def __getitem__(self, index):
        # Return processed image and label

        if self.mode == 'train':
            path = os.path.join(self.root, "iclevr", self.img_name[index])
            transform=transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
            img = transform(Image.open(path).convert('RGB'))
            #print(img.shape)
        else:
            img = torch.ones(1) # for sampling, give a dummy values 

        label = self.label[index]
        return img, label


    