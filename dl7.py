###CNN(CIFAR10 DATASET)---->Pytorch tutorial

###Training a classifier
###1.Load and normalize the CIFAR10 training and test datasets using torchvision
###2.Define a Convolutional Neural Network
###3.Define a loss function
###4.Train the network on the training data
###5.Test the network on the test data

###1.
import torch
import torchvision
import torchvision.transforms as transforms

###Converts the images from PIL (Python Imaging Library) images or NumPy arrays to PyTorch tensors.
###The pixel values are scaled to be between 0 and 1.
###Normalizes the image tensor. The first tuple specifies the mean for each channel (R, G, B), and the second tuple specifies the
###standard deviation. Here, the values are normalized to a range of approximately -1 to 1.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4 ###the data will be processed in mini-batches of 4 images at a time.

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
###torch.utils.data.DataLoader: A PyTorch utility that loads data in mini-batches.num_workers=2-->The number of subprocesses to use
###for data loading. More workers can speed up the data loading process.

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###showing some of the training images for fun
import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    img = img / 2 + 0.5     ###unnormalize
    npimg = img.numpy()     ###tensor to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  ###The image data is transposed from (C, H, W) format (channels, height, width) to (H, W, C) format (height, width, channels) so that it can be correctly displayed by plt.imshow
    plt.show()
###get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
###show images
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
###labels[j]: Fetches the label for the jth image in the batch.
###classes[labels[j]]: Converts the label (which is an integer index) into the corresponding class name.
###f'{classes[labels[j]]:5s}': Formats the class name to be left-justified within a field of 5 characters.
###' '.join(...): Joins all the formatted class names into a single string, separated by spaces.


###2.
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        ###self.conv1 = nn.Conv2d(3, 6, 5):
        ###This is a 2D convolutional layer.
        ###3: Number of input channels (for RGB images, this is 3).
        ###6: Number of output channels (or feature maps) produced by the convolution.
        ###5: Size of the convolutional kernel (5x5).
        ###self.pool = nn.MaxPool2d(2, 2):
        ###This is a max pooling layer.
        ###2, 2: The kernel size and stride are both 2x2, which means the output size is reduced by half in both spatial dimensions (height and width).
        ###self.conv2 = nn.Conv2d(6, 16, 5):
        ###Another 2D convolutional layer.
        ###6: Number of input channels (the output from conv1).
        ###16: Number of output channels (or feature maps).
        ###5: Size of the convolutional kernel (5x5).
        ###self.fc1 = nn.Linear(16 * 5 * 5, 120):
        ###A fully connected (dense) layer.
        ###16 * 5 * 5: The number of input features, calculated from the output of the last convolutional layer after pooling.
        ###120: The number of output features.
        ###self.fc2 = nn.Linear(120, 84):
        ###Another fully connected layer.
        ###120: Number of input features (output from fc1).
        ###84: Number of output features.
        ###self.fc3 = nn.Linear(84, 10):
        ###The final fully connected layer.
        ###84: Number of input features (output from fc2).
        ###10: Number of output features, corresponding to the number of classes in the CIFAR-10 dataset.
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()


###3.
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
###This is a commonly used loss function for classification problems, especially when the output classes are mutually
###exclusive (like in CIFAR-10).
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


###4.
for epoch in range(2):
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
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')
### the network is trained on the training data for two epochs. During each epoch:
###The network processes mini-batches of data, computes the loss, and adjusts its parameters to reduce this loss.
###Every 2000 mini-batches, it prints the average loss to monitor training progress.


###5.
dataiter = iter(testloader)
images, labels = next(dataiter)
###print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

