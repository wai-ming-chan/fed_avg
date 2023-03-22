#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/wai-ming-chan/fed_avg/blob/main/sanityCheck_MNIST.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Demo program to train MNIST with Pytorch (Centralized training)
# 
# Reference: [https://meinkappa.github.io/blog/2021/09/16/MNIST-In-Pytorch.html](https://meinkappa.github.io/blog/2021/09/16/MNIST-In-Pytorch.html)

# In[1]:


# import libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------------------------
# libraries for pyTorch model Visualization
get_ipython().system(' pip install -q torchview')
get_ipython().system(' pip install -q -U graphviz')

import torchvision

from torchview import draw_graph
from torch import nn
import torch
import graphviz

# when running on VSCode run the below command
# svg format on vscode does not give desired result
graphviz.set_jupyter_format('png')
#-----------------------------------------------------------------------
get_ipython().system(' pip install -q torch-summary')


# ### Loading dataset
# 

# In[2]:


# grab MNIST data with torchvision datasets
## We can tell Pytorch how to manipulate the dataset by giving details.
##
### root: Where to store the data. We are storing it in data directory.
### train: Whether to grab training dataset or testing dataset. 
###         Given True value, training_data is a training dataset from MNIST. 
###         On the other hand, test_data is a testing dataset from MNIST.
### download: Whether to download if data is not already in root. We passed True to download the dataset.
### transform: What to do with data. We are converting our images of handwritten digits into Pytorch tensors so that we can train our model.

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# Set GPU or CPU device. If GPU is available, we use it to speed up the training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('[GPU mode] device: ', device)
else:
    print('[CPU mode] device: ', device)
    

# check the dimension of data
print(training_data)
print('training data size: ', len(training_data) )
print('test data size: ', len(test_data) )
print(training_data[0][0].shape)
print(training_data[0][0].squeeze().shape)
# plt.imshow(training_data[len(training_data)-1][0].squeeze(), cmap="gray");
# print('label: ', training_data[len(training_data)-1][1])


figure = plt.figure(figsize=(6,3))
cols, rows = 4, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# In[3]:


# create a dataloader. 
## A dataloader divides our data by a given batch_size and hands out each one to our model for training
## Our train_dataloader will have 64 images per batch, which makes a total of 157 batches.
bs=64
train_dataloader = DataLoader(training_data, batch_size=bs)
test_dataloader = DataLoader(test_data, batch_size=bs)

print(10000/64, len(test_dataloader))


# In[4]:


# setup our model now.

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Feed our model into GPU now if there is one.
model = NeuralNetwork().to(device)
# print(model)

from torchsummary import summary
# help(summary)
# summary(model, input_size=(1,784), batch_size=bs)


# Model 1: 2NN
class MNIST_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model_2NN = MNIST_2NN().to(device)
summary(model_2NN, input_size=(1,784), batch_size=bs)


model_graph = draw_graph(
  model_2NN, 
  input_size=(bs,784), 
  graph_name='my2NN',
  hide_inner_tensors=True,
  hide_module_functions=False,
  expand_nested=True,
  roll=True, # rolls recursive models
  save_graph=True
)
model_graph.visual_graph
#-----------------------------------------------------------------------


# ### Visualization of our defined models

# In[5]:


if False:
  # Visualization of model and trace (method 1: Torchviz )
  ## ref: https://github.com/szagoruyko/pytorchviz
  get_ipython().system('pip install torchviz')
  from torchviz import make_dot

  # model = nn.Sequential()
  # model.add_module('W0', nn.Linear(8, 16))
  # model.add_module('tanh', nn.Tanh())
  # model.add_module('W1', nn.Linear(16, 1))
  # make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

  # x = torch.randn(1, 8)
  # y = model(x)
  # print('x: ', x)
  # print('y: ', y)

  model = NeuralNetwork().to(device)
  sample_idx = torch.randint(len(training_data), size=(1,)).item() # randomly pick one sample
  img, label = training_data[sample_idx]
  y = model(img)
  make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)


# In[6]:


if False:
  # Visualization of pyTorch model (method 2: torchview)
  # source: https://github.com/mert-kurttutan/torchview
  get_ipython().system(' pip install -q torchview')
  get_ipython().system(' pip install -q -U graphviz')

  import torchvision

  from torchview import draw_graph
  from torch import nn
  import torch
  import graphviz

  # when running on VSCode run the below command
  # svg format on vscode does not give desired result
  graphviz.set_jupyter_format('png')

  summary(model, input_size=(1,784), batch_size=64)

  batch_size=64
  model_graph = draw_graph(
    model, 
    input_size=(batch_size,784), 
    graph_name='myNN',
    hide_inner_tensors=True,
    hide_module_functions=False,
    expand_nested=True,
    roll=True, # rolls recursive models
    save_graph=True
  )
  model_graph.visual_graph


# ### Some model hyperparameters setting

# In[7]:


# model parameters setting (will affect performance of the model)
lr = 1e-3     # learning rate
bs = 64       # batch size
epochs = 500    # more epochs the more likely our model is going to learn from the dataset. However, too many epochs will overfit our model

loss_fn = nn.CrossEntropyLoss() # Classificatin problem, we use Cross Entropy as loss function

optimizer = torch.optim.SGD(model_2NN.parameters(), lr=lr) # an optimizer to update our parameters, we use stochastic gradient descent to reduce the loss.

# define how we train our data
def train_data(model):
    for xb, yb in train_dataloader:   # fetch xb (a batch of images) and yb (a batch of labels) from train_dataloader
        if device == 'cuda':
            xb, yb = xb.cuda(), yb.cuda()
        preds = model(xb)             # make a prediction
        loss = loss_fn(preds, yb)     # compute the loss value
        optimizer.zero_grad()         # set our gradient to zero
        loss.backward()               # gradient descent
        optimizer.step()              # update our weights and biases
    loss = loss.item()
    # print(f"Train loss: {loss:>7f}")
    return loss



# define how we test our model
def test_data(model):
    num_batches = len(test_dataloader)
    size = len(test_dataloader.dataset)
    test_loss, corrects = 0, 0

    with torch.no_grad():             # Disabling gradient calculation for reduction of memory consumption for computations
        for xb, yb in test_dataloader:
            if device == 'cuda':
                xb, yb = xb.cuda(), yb.cuda()
            preds = model(xb)
            test_loss += loss_fn(preds, yb).item()
            corrects += (preds.argmax(1) == yb).type(torch.float).sum().item()
  
    test_loss /= num_batches
    corrects /= size
    # print(f"Test loss: \n Accuracy: {(100*corrects):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return corrects, test_loss

lossTR_vals= []
lossTT_vals= []
accu_vals= []
for t in range(epochs):
    loss_train_t = train_data(model_2NN)
    accu_t, loss_test_t = test_data(model_2NN)
    lossTR_vals.append(loss_train_t)
    lossTT_vals.append(loss_test_t)
    accu_vals.append(accu_t)

    if t % 10 == 0:
        print(f'[epoch: {t}\t Train loss: {loss_train_t:>7f}, Accuracy: {(100*accu_t):>0.1f}%, Test loss: {loss_test_t:>8f} ')

# plt.plot(np.linspace(1, epochs,epochs).astype(int), loss_vals)
# plt.plot(np.linspace(1, epochs,epochs).astype(int), loss_vals)

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(lossTT_vals,label="val")
plt.plot(lossTR_vals,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Loss.png')
plt.show()

plt.figure(figsize=(10,5))
plt.title("Test Data Accuracy")
plt.plot(accu_vals)
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.savefig('Accuracy.png')
# plt.legend()
plt.show()


# In[ ]:




