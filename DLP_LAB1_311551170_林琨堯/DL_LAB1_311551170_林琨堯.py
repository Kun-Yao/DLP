# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:36:14 2023
https://github.com/csielee/2019DL/blob/master/lab1/lab1.py
https://github.com/chiha8888/NCTU_DLP/blob/master/lab1/lab1_withbias.py
https://github.com/steven112163/Deep-Learning-and-Practice/blob/main/Lab%201/backpropagation.py
@author: Lin
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

      
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

class Net:
    def __init__(self, lr, epoch, hidden_size):
        self.w=[None, np.random.randn(hidden_size[0],2),np.random.randn(hidden_size[1],hidden_size[0]),np.random.randn(1,hidden_size[1])]
        self.b=[None, np.zeros((hidden_size[0],1)),np.zeros((hidden_size[1],1)),np.zeros((1,1))]
        self.l=[None, np.zeros((hidden_size[0],1)),np.zeros((hidden_size[1],1)),np.zeros((1,1))]
        self.s=[None, np.zeros((hidden_size[0],1)),np.zeros((hidden_size[1],1)),np.zeros((1,1))]
        
    def get_loss(self, y, y_hat):
        return np.mean((y-y_hat)**2)
    
    def forward_pass(self, x):
        #forward pass
        self.s[0] = x
        self.l[1] = self.w[1].dot(self.s[0])
        self.s[1] = sigmoid(self.l[1])
        self.l[2] = self.w[2].dot(self.s[1])
        self.s[2] = sigmoid(self.l[2])
        self.l[3] = self.w[3].dot(self.s[2])
        self.s[3] = sigmoid(self.l[3]) #y_hat
        
        return self.s[3]
    
    def backward_pass(self, y, y_hat):
        #backward pass
        g_s3 = (2) * (y_hat - y) / y.shape[1]
        g_l3 = g_s3*derivative_sigmoid(self.s[3])
        g_w3 = g_l3.dot(self.s[2].transpose())
        g_b3 = np.sum(g_l3,axis=1,keepdims=True) * (1/y.shape[0])
        
        g_s2 = ((self.w[3].transpose()).dot(g_l3))
        g_l2 = g_s2*derivative_sigmoid(self.s[2])
        g_w2 = g_l2.dot(self.s[1].transpose())
        g_b2 = np.sum(g_l2,axis=1,keepdims=True) * (1/y.shape[0])
            
        g_s1 = ((self.w[2].transpose()).dot(g_l2))
        g_l1 = g_s1*derivative_sigmoid(self.s[1])
        g_w1 = g_l1.dot(self.s[0].transpose())
        g_b1 = np.sum(g_l1,axis=1,keepdims=True) * (1/y.shape[0])
        
        #update weight
        self.w[1] -= lr*g_w1
        self.w[2] -= lr*g_w2
        self.w[3] -= lr*g_w3
        self.b[1] -= lr*g_b1
        self.b[2] -= lr*g_b2
        self.b[3] -= lr*g_b3
                        
    def visualize(self,x, y, pred_y):
        plt.subplot(1,2,1)
        plt.title("Groundtruth")
        for i in range(x.shape[1]):
            if y[0,i] == 1:
                plt.plot(x[0,i], x[1,i], 'b.')
            else:
                plt.plot(x[0,i], x[1,i], 'r.')
                
        plt.subplot(1,2,2)
        plt.title("Prediction")
        for i in range(x.shape[1]):
            if pred_y[0,i] == 1:
                plt.plot(x[0,i], x[1,i], 'b.')
            else:
                plt.plot(x[0,i], x[1,i], 'r.')
        plt.show()    
 #%%   
x, y = generate_linear(n=100)
x = x.transpose()
y = y.transpose()

hidden_size = [15,10]
y_hat = np.zeros(y.shape)  
Epoch = []

epoch = 5000
lr = 0.06
loss = np.zeros((epoch,1))

linear_case = Net(lr, epoch, hidden_size)
#train
for i in range(epoch):
    y_hat = linear_case.forward_pass(x)
            
    loss[i] = linear_case.get_loss(y, y_hat)
    if i % 500 == 0:
        print("epoch {}: loss={}" .format(i, loss[i])) 
    
    linear_case.backward_pass(y, y_hat)
    
    Epoch.append(i)
print("epoch {}: loss={}" .format(i, loss[i-1])) 
   
plt.plot(Epoch, loss)
plt.title("Linear")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()  
     
#predict
pred_y = [0]*x.shape[1]
pred_y = linear_case.forward_pass(x)
print(pred_y)


ry = np.round(pred_y)
linear_test_loss = linear_case.get_loss(y, ry)
linear_test_acc = 1.0-(np.sum(np.abs(y-ry))/y.shape[1])
linear_case.visualize(x, y, ry)


#%%
x, y = generate_XOR_easy()
x = x.transpose()
y = y.transpose()

hidden_size = [15,10]
y_hat = np.zeros(y.shape)  
Epoch = []

epoch = 20000
lr = 0.08
loss = np.zeros((epoch,1))

xor_case = Net(lr, epoch, hidden_size)
#train
for i in range(epoch):
    y_hat = xor_case.forward_pass(x)
            
    loss[i] = xor_case.get_loss(y, y_hat)
    if i % 2000 == 0:
        print("epoch {}: loss={}" .format(i, loss[i])) 
    
    xor_case.backward_pass(y, y_hat)
    
    Epoch.append(i)
print("epoch {}: loss={}" .format(i, loss[i]))    
plt.plot(Epoch, loss)
plt.title("XOR")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()  
     
#predict
pred_y = [0]*len(x)
pred_y = xor_case.forward_pass(x)
print(pred_y)


ry = np.round(pred_y)
xor_test_loss = xor_case.get_loss(y, y_hat)
xor_test_acc = 1.0-(np.sum(np.abs(y-ry))/y.shape[1])
xor_case.visualize(x, y, ry)
#%%
print("linear_test_loss={}" .format(linear_test_loss))
print("linear_test_acc={}" .format(linear_test_acc))
print("xor_test_loss={}" .format(xor_test_loss))
print("xor_test_acc={}" .format(xor_test_acc))
