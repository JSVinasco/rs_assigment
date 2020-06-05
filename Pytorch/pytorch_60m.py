########################################
###     Pytorch 60 minutes Tutorial  ###
########################################

# Pytorch is a library with two main funcionalities
# the firts one is a scientific computing library with GPU support
# and the second is a neuronal network library
# In the next code i follow the pytorch 60 minutes tutorial

# import the libraries


from __future__ import print_function
import torch


import os
#os.chdir('/home/juan/Documentos/cesbio/assigments/Code/')
os.chdir('E:/CESBIO/preparation/rs_assigment/Pytorch/')

###################################
###     Part 1 : Torch Tensors  ###
###################################

# in the next lines is define the sintaxis 
# and behavior of the pytorch tensor 
# a replacement of numpy ndarray's

# create a matrix without initialized 
# with 5 rows and 3 columns
x = torch.empty(5, 3)
print(x)
# create a random matrix 
# with 5 rows and 3 columns
x = torch.rand(5, 3)
print(x)
# create zeros matrix 
# with 5 rows and 3 columns
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
# input the data manually
x = torch.tensor([5.5, 3])
print(x)
# create zeros matrix with a double data type
# with 5 rows and 3 columns
x = x.new_ones(5, 3, dtype=torch.double)      
print(x)
# create random matrix with a float data type
# with 5 rows and 3 columns
x = torch.randn_like(x, dtype=torch.float)    
print(x)                                      
# verify the matrix size
print(x.size())
###################
#### Operations ###
###################
# sum of matrix
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# using a predefine tensor to store the result
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# adds x to y
y.add_(x)
print(y)
# slicing the tensor witha numpy like sintaxis 
print(x[:, 1])
# resizing the tensor this comand change the 
# key worth reshape for view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
# is possibly use .item() to get the value as python number
# working for a scalar data not with tensors
x = torch.randn(1)
print(x)
print(x.item())
#####################
#### NumPy Bridge ###
#####################
# Is possibly transfor pytorch tensor 
# to numpy ndarray and vice versa
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA Tensors
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

####################################
###     Part 2 : Torch Autograd  ###
####################################
#The next module is the autograd (automatic differentiation)
#this package allow the differentiation operations for the tensors
#addicionally track this operation, for the train block of a neural net
#some important comand is .requires_grad as True for start the 
#operational tracking, .backward() for compute the gradients
#.detach() to stop the tracking and prevent the computation of 
#future differention calculations, with torch.no_grad(): for 
#prevent modification of the weights of the neural net are updated
#during the prediction block, and finally the  Function atributte of
#the different functions of the pytorch library, and allow compute the
#differention.
#

# create a tensor with the tracking activate
x = torch.ones(2, 2, requires_grad=True)
print(x)
# use that tensor for a operation
y = x + 2
print(y)
# show the pointer that stores the track of operations
print(y.grad_fn)
# applied a operation for stores the data for the gradients
z = y * y * 3
out = z.mean()
print(z, out)
# example of how activate o desactivate the gradient track
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
# compute a operation for store the tracking parents tensor
# and applied operations
b = (a * a).sum()
print(b.grad_fn)
##################
#### Gradients ###
##################
# propagate the gradients 
out.backward()
print(x.grad)
# example of propagate the gradients 
# for compute the jacobian-vector product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
# create a vector with the same size of y
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# calculate the gradient of y respect to v and store the information on y
y.backward(v)
# the gradient are store in the .grad propiertys
print(x.grad)
#
print(x.requires_grad)
print((x ** 2).requires_grad)
# also is posible stop the tracking of the parameters
with torch.no_grad():
    print((x ** 2).requires_grad)
# for that is possible is the detach command like next
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())





###########################################
###     Part 3 : Torch Neural Networks  ###
###########################################
# for construct a neural nets in pytorch exist the torch.nn module
# is depend of the autograd module and focus in construct the architecture
# and parameters of the neural net.
# the tipic procedure is the next:
#Define the neural network that has some learnable parameters (or weights)
#Iterate over a dataset of inputs
#Process input through the network
#Compute the loss (how far is the output from being correct)
#Propagate gradients back into the network’s parameters
#Update the weights of the network, typically using a simple update rule

#########################
#### Define a Network ###
#########################
# the importing of libraries is the tensor, neuralnets and Functional 
# modules

import torch
import torch.nn as nn
import torch.nn.functional as F

# Firts we define the neurnal net using class net who extend the nn.Module
# Is precise define in the init the architecture 
# In the foward we define the non-linear functions

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
            return num_features


net = Net()
print(net)

# addicionally is necesary define some parametes like the zero value of
# the gradient, and the backward.
        
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

input = torch.rand(1, 1, 32, 32)
input2 = torch.reshape(input,(32,32,1,1))
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.rand(1,10))

######################
#### Loss Function ###
######################

# another importan part is the loss function define in pytorch
# like a 'criterion' 

output = net(input)
target = torch.rand(10) # a dummy target
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)



print(loss.grad_fn) #MSELoss
print(loss.grad_fn.next_functions[0[0]]) # Linear
print(loss.grad_fn.next_function[0][0].next_function[0][0]) # ReLU

########################
#### Backpropagation ###
########################
# the backpropagation algorith let update the weights of the neural net
# based in the loss computed in the last step

net.zero_grad() # zeroes the gradient buffers og all parameters

print('conv1.bias.grad before backard')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backard')
print(net.conv1.bias.grad)

###########################
#### Update the weights ###
###########################

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# create your optimizer

optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step()


###############################################
###     Part 4 : Torch Training Classifier  ###
###############################################

#sera que si

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

inputs, labels = data[0].to(device), data[1].to(device)

