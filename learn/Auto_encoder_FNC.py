
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

#from torchinfo import summary

# Hyperparameters

def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.reshape(-1, 28*28)
        # Compute prediction error
        pred = model(X).to(device)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 8 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  {(current/size*100):>2f}%")

def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cuda:0" ), y.to("cuda:0" )
            X = X.reshape(-1, 28*28)
            pred = model(X).to(device)
            test_loss += loss_fn(pred, X).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return X,pred


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

    training_data_load = DataLoader(training_data,batch_size=64,shuffle = True)
    
    test_data_load = DataLoader(test_data,batch_size=1000,shuffle = True)

    return training_data_load,test_data_load

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
                                nn.Linear(784,512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512,256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,128),
                                nn.ReLU(),
                                nn.Linear(128,10),
                                )
        self.code = nn.Linear(10,10)
        self.decoder = nn.Sequential(nn.Linear(10,128),
                                     nn.ReLU(),
                                     nn.Linear(128,256),
                                     nn.ReLU(),
                                     nn.Linear(256,512),
                                     nn.ReLU(),
                                     nn.Linear(512,784),
                                     nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.code(x)
        x = self.decoder(x)
        return x




def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    learning_rate = 0.001
    num_epochs = 30
    batch_size = 32
    
    CNN = Net().to(device)
    train_data,testing_data = Lunchdata()

    for X, y in testing_data:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
   
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate,weight_decay = 1e-8)
    #summary(Net().to(device),input_size=(batch_size,1,28,28),dtypes=[torch.float, torch.long])
    
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data, CNN, loss_fn, optimizer,device)
       
    input,output = test(testing_data, CNN, loss_fn,device)
    input = input.to("cpu")
    output = output.to("cpu")
    print("Done!")
    print('Finished Training')

    with torch.no_grad():
        number = 20
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(input[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(output[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    
    
if __name__ == "__main__":
        main()