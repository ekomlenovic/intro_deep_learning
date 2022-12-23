
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torchinfo import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)  
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.conv3 = nn.Conv2d(64,128,3,1)
        self.conv4 = nn.Conv2d(128,256,3,1)
        
        self.Dropout1 = nn.Dropout(0.18)
        self.Dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        
        self.decoder1 = nn.Linear(10,128)
        self.decoder2 = nn.Linear(128,256)
        self.decoder3 = nn.Linear(256,512)
        self.decoder4 = nn.Linear(512,28*28)

    def encode(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.Dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.Dropout1(x)
            
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = self.Dropout2(x)
        x = self.fc3(x)
        return x    
    
    def decode(self, x):

        x = F.relu(self.decoder1(x))
        x = F.relu(self.decoder2(x))
        x = F.relu(self.decoder3(x))
        x = F.sigmoid(self.decoder4(x))
        
        return x

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)

        return x



def main():
    
    test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        ]),
    
    )
    
    test_data_load = DataLoader(test_data,batch_size=10000,shuffle = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNN = torch.load("../data/Model_learn/MNIST_auto_encodeur.pth",device)

    tensn1 = torch.tensor((),dtype=torch.float64)
    tensn1 = tensn1.new_zeros(10)
    tensn2 = torch.tensor((),dtype=torch.float64)
    tensn2 = tensn2.new_zeros(10)
    n1 = int(input('Donner Un chiffre : \n'))
    n2 = int(input('Donner Un autre chiffre : \n'))
    index = 0
    
    for _,(x,y) in enumerate(test_data_load):
        for i in y:
            if i == n1:
                tensn1 = CNN.encode(x[index].to(device))
                index = 0
                break
            index +=1
        print(tensn1)
        for i in y:
            if i == n2:
                tensn2 = CNN.encode(x[index].to(device))
                index = 0
                break
            index +=1
        print(tensn2)



    for i in range(0,11):
        tensn12 = (tensn1*(1-i/10) + tensn2*(i/10))
        tensn12 = CNN.decode(tensn12)
        tensn12 = tensn12.to('cpu').detach().numpy()
    
        ax = plt.subplot(1, 11,  i+1)
        plt.imshow(tensn12.reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
if __name__ == "__main__":
        main()