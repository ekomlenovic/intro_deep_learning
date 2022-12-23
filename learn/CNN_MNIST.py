
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tensorflow as tf
import datetime

from torchinfo import summary


def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset) #calcule la taille du jeu de donnée d'entrainement
    for batch, (X, y) in enumerate(dataloader):     #donne l'index de l'image X=tensor de l'image de taille 28*28 de la taille du bash chargé 
                                                    # et y = tensor de label
        X, y = X.to(device), y.to(device) #On charge les tensors dans le gpu ou dans le cpu

        # Compute prediction error
        pred = model(X) #On prédit y
        loss = loss_fn(pred, y) #On regarde le loss

        # Backpropagation optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 64 == 0: 
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  {(current/size*100):>2f}%")

def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def Lunchdata():
    
    training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor(),
        )   


    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
        )

    training_data_load = DataLoader(training_data,batch_size=16,shuffle = True)
    
    test_data_load = DataLoader(test_data,batch_size=10000,shuffle = True)

    return training_data_load,test_data_load

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)  
        self.conv2 = nn.Conv2d(32,32,3,1)
        self.conv3 = nn.Conv2d(32,64,3,1)
        self.conv4 = nn.Conv2d(64,64,3,1)
        
        self.Dropout1 = nn.Dropout(0.18)
        self.Dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.Dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.Dropout1(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.Dropout2(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x




def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    learning_rate = 0.001
    num_epochs = 15
    batch_size = 32
    
    CNN = Net().to(device)
    train_data,testing_data = Lunchdata()

    for X, y in testing_data:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
   
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(CNN.parameters(), lr=learning_rate,momentum=0.90)
    summary(Net().to(device),input_size=(1,1,28,28),dtypes=[torch.float, torch.long])
    
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data, CNN, loss_fn, optimizer,device)
        test(testing_data, CNN, loss_fn,device)
        
    torch.save(CNN.state_dict(), '../data/Model_learn/CNN_MNIST.pth')
    print("Done!")
    print('Finished Training')

    
if __name__ == "__main__":
        main()