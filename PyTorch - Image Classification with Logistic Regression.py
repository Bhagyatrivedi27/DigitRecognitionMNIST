#!/usr/bin/env python
# coding: utf-8

# # Image Classification with Logistic Regression
# MNIST Handwritten Digits Database

# # Exploring the data
# We will import torch, torchvision. It contains some utilities for working with image data. It also contains helper classes to automatically download and import popular datasets like MNIST

# In[3]:


#imports
import torch
import torchvision
from torchvision.datasets import MNIST


# In[4]:


#Download training dataset
dataset = MNIST(root='data/', download = True)


# In[5]:


len(dataset)


# Additional 10,000 images of test set can be created by passing train = Fase to MNIST class

# In[6]:


test_dataset = MNIST(root='data/', train = False)
len(test_dataset)


# In[7]:


dataset[0]


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:',label)


# In[10]:


image, label = dataset[10]
plt.imshow(image, cmap='gray')
print('Label:', label)


# We need to convert these images into tensors. We can use transforms

# In[11]:


import torchvision.transforms as transforms


# In[12]:


#MNIST dataset (images and labels)
dataset = MNIST(root='data/', train=True, transform= transforms.ToTensor())


# In[13]:


img_tensor, label = dataset[0]
print(img_tensor.shape, label)


# The image is now converted to a 1x28x28 tensor. The first dimension is used to keep track of color channels. Since images in MNIST are grayscale, there's just one channel. OTher datasets have images with color, in which case there are 3 channels: red, green and blue (RGB). 

# In[16]:


print(img_tensor[:,10:15,10:15])
print(torch.max(img_tensor), torch.min(img_tensor))


# In[17]:


#plot the image by passing in the 28x28 matrix
plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')


# # Training and Validation Datasets
# while building real world machine learning models, it is quite common to splite the dataset into 3 parts:
# - Training set: used to train the model i:e. compute the loss and adjust the weights of the model using gradient descent
# - Validation set: used to evaluate the model while training, adjust hyperparams (learning rate, etc) and pick the best version of the model
# - Test set: used to compare different models or different types of modelling approaches and report the final accuracy of the model 

# In[18]:


import numpy as np 

def split_indices(n, val_pct):
    #determine size of validation srt 
    n_val = int(val_pct*n)
    #Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    #Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


# split indices randomly shuffles the array indices 0,1,..,n-1 and separates out a desired portion from it for the validation set. 

# In[19]:


train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)


# In[20]:


print(len(train_indices), len(val_indices))
print('Sample val indices:', val_indices[:20])


# In[21]:


from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


# In[28]:


batch_size = 100

#Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size, sampler = train_sampler)

#Validation sampler and data loader 
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size, sampler = val_sampler)


# # Model
# 
# - A logistic Regression model is almost identical to a linear regression model i:e, there are weights and bias matrices and the output is obtained using simple matrix operations (pred = x @ w.t() + b)
# 
# - Just as we did the linear regression we can use nn.Linear to create the model instead of defining and initializing the matrices manually
# 
# - Since nn.Linear expects the each training example to be a vector, each 1x28x28 image tensor needs to be flattered out into a vector of size 784(28*28), before being passed into the model
# 
# - The output for each image is vector of size 10 with each element of the vector signifying the probability a particular target label (0 to 9). The predicted label for an image is simply the one with the highest probability

# In[29]:


import torch.nn as nn

input_size = 28*28
num_classes = 10

#logistic regression model 
model = nn.Linear(input_size, num_classes)


# This model is a lot larger than prev mode, in terms of number of params

# In[30]:


print(model.weight.shape)
model.weight


# In[31]:


print(model.bias.shape)
model.bias


# In[33]:


for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images)
    break


# We will use .reshape method of a tensor which will aloow us to efficiently 'view' each image as a flat vector, without really changing the underlying data
# 
# To include this additional functionality within our model, we need to define a custom model, by extending the nn.module class from PyTorch

# In[34]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()


# Inside the __init__ constructor method, we instantiate the weights and biases using nn.Linear. And inside the forward method, which is invoked when we pass a batch of inputs to the model, we flatten out the input tensor and then pass it into self.linear
# 
# xb.reshape(-1, 28*28) indicates to PyTorch that we want a view of the xb tensor with two dimensions where the length along the 2nd dimension is 28*28 (i:e 784). One arguement to .reshape can be set to -1 (in this case the first dimension), to let PyTorch figure it out automatically based on the shape of the original tensor

# Now weight and bias is under model.linear.weight and model.linear.bias

# In[36]:


print(model.linear.weight.shape, model.linear.bias.shape)
list(model.parameters())


# In[38]:


for images, labels in train_loader:
    outputs = model(images)
    break
    
print('outputs.shape :', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)


# To convert the output rows into probabilities, we use the softmax function 

# In[39]:


import torch.nn.functional as F


# In[40]:


#Apply softmax for each output row
probs = F.softmax(outputs, dim=1)

#Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

#Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())


# Finally we can determine the predicted label for each image by simply choosing the index of element with highest probability in each output row. This is done using torch.max, which returns the largest element and the index of the largest element along a particular dimension of a tensor

# In[54]:


max_probs, preds = torch.max(probs, dim=1)
print(preds)


# In[55]:


labels


# # Evaluation Metric and Loss function
# 
# Just as with linear regression, we need a way to evaluate how well our model is performing. A natural way to do this would be to find the percentage of labels that were predicted correctly, i:e. the accuracy of predictions.

# In[56]:


torch.sum(preds == labels).item()/len(preds)


# In[57]:


def accuracy(l1, l2):
    return torch.sum(l1==l2).item() / len(l1)


# In[59]:


accuracy(preds, labels)


# Cross entropy is a continuous and differential fn that also provides good feedback for incremental improvementsin model. 

# In[60]:


loss_fn = F.cross_entropy


# In[61]:


loss = loss_fn(outputs, labels)
print(loss)


# # Optimizer

# In[62]:


learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# In[63]:


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    #calculate loss
    preds = model(xb)
    loss = loss_func(preds, yb)
    
    if opt is not None:
        #compute gradients 
        loss.backward()
        #Update parameters 
        opt.step()
        #Reset gradients
        opt.zero_grad()
        
    metric_result = None
    if metric is not None:
        #Compute the metric
        metric_result = metric(preds, yb)
        
    return loss.item(), len(xb), metric_result


# In[65]:


def evaluate(model, loss_fn, valid_dl, metric= None):
    with torch.no_grad():
        #pass each batch through the model 
        results = [loss_batch(model, loss_fn, xb, yb, metric = metric)
                  for xb, yb in valid_dl]
        #Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        
        #total size of the dataset
        total = np.sum(nums)
        
        #Avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            #Avg of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
            
    return avg_loss, total,avg_metric


# In[66]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


# In[67]:


val_loss, total, val_acc = evaluate(model, loss_fn, val_loader, metric = accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))


# In[68]:


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    for epoch in range(epochs):
        #Training
        for xb, yb in train_dl:
            loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)
            
        #Evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result
        
        #Print progress 
        if metric is None: 
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric ))


# In[69]:


#Rededine model and optimizer 
model = MnistModel()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# In[71]:


fit(5, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)


# In[72]:


fit(5, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)


# In[74]:


fit(5, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)


# In[75]:


fit(5, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)


# In[76]:


accuracies = [0.6402,0.7335,0.7733,0.7955,0.8088,0.8177,0.8233,0.8297,0.8337,0.8372,0.8395,0.8424,0.8445,0.8470,0.8488,0.8510,0.8522,0.8543,0.8551,0.8566]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of Epochs');


# # Testing on images
# 

# In[77]:


#define test dataset
test_dataset = MNIST(root= 'data/', train=False, transform= transforms.ToTensor())


# In[78]:


img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label)


# In[79]:


img.unsqueeze(0).shape


# In[80]:


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


# In[82]:


img, label= test_dataset[1222]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img,model))


# In[83]:


torch.save(model.state_dict(), 'mnist-logistic.pth')


# In[84]:


model.state_dict()


# In[85]:


model2 = MnistModel()
model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()


# In[ ]:




