
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

from torch.utils.tensorboard import SummaryWriter

from scipy import *
from torchinfo import summary

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset) #calcule la taille du jeu de donnée d'entrainement
    test_loss = []
    correct =[]
    for batch, (X, y) in enumerate(dataloader):     #donne l'index de l'image X=tensor de l'image de taille 28*28 de la taille du bash chargé 
                                                    # et y = tensor de label
        X, y = X.to(device), y.to(device) #On charge les tensors dans le gpu ou dans le cpu

        # Compute prediction error
        pred = model(X) #On prédit y
        X=X.reshape(-1, 28*28)
        loss = loss_fn(pred, X) #On regarde le loss

        # Backpropagation optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 64 == 0: 
            loss, current = loss.item(), batch * len(X)
            test_loss.append(loss_fn(pred, X).item())
            correct.append( (pred.argmax(1) == y).type(torch.float).sum().item())
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  {(current/size*100):>2f}%")
    return test_loss, correct

def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    tab_loss , tab_correct = [] ,[]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            X=X.reshape(-1, 28*28)
            test_loss += loss_fn(pred, X).item()
            tab_loss.append(test_loss)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            tab_correct.append(correct)
            print(f"Test Error: \n Accuracy: {(correct):>0.1f}, Avg loss: {test_loss:>8f} \n")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return X,y,pred,tab_loss,tab_correct


def Lunchdata():
    
    

    training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        ]),
    )

    training_data_rot  = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(45),
        ]),
    )


    test_data_rot  = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(45),
        ]),
    )


    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            ]),
    )

    test_data_bruit = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            AddGaussianNoise(0.25, 0.5)
            ]),
    )

    test_data_flou = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.GaussianBlur(kernel_size = (3,3), sigma=(1, 10))
            ]),
    )
    

    training_data_load = DataLoader(training_data,batch_size=64,shuffle = True)
    training_data_load_rot = DataLoader(training_data_rot,batch_size=64,shuffle = True)
    
    test_data_load = DataLoader(test_data,batch_size=10000,shuffle = True)
    test_data_load_rot = DataLoader(test_data_rot,batch_size=10000,shuffle = True)
    test_data_load_bruit = DataLoader(test_data_bruit,batch_size=10000,shuffle = True)
    test_data_load_flou = DataLoader(test_data_flou,batch_size=10000,shuffle = True)

    return training_data_load,training_data_load_rot,test_data_load,test_data_load_bruit,test_data_load_rot,test_data_load_flou

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
            
        x = torch.flatten(x,1)
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
    learning_rate = 0.001
    num_epochs = 2
    o=0
    batch_size = 32
    train_data,train_data_rot,testing_data,testing_data_bruit,testing_data_rot,testing_data_flou = Lunchdata()
    t_lose = []
    test_lose = []
    t_correct =[]
    test_correct =[]
    writer = SummaryWriter("../data/MNIST")


    for X, y in testing_data:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    loss_fn = nn.MSELoss()
    # Device
    i= input("voulez vous réentrainer le model ? y/n" + '\n')
    if i == "n":
        CNN = torch.load("../data/Model_learn/MNIST_auto_encodeur.pth",device)
    else :
        CNN = Net().to(device)
        optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate,weight_decay=1e-8)

        summary(Net().to(device),input_size=(64,1,28,28),dtypes=[torch.float, torch.long])
        
        for t in range(num_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            a,b  =train(train_data, CNN, loss_fn, optimizer,device)
            t_lose.append(a)
            t_correct.append(b)
            #train(train_data_rot, CNN, loss_fn, optimizer,device)
        torch.save(CNN, '../data/Model_learn/MNIST_auto_encodeur.pth')
    
    for n_iter in t_lose:
        for t in n_iter:
            writer.add_scalar('Loss/train', t, o)
            o+=1

    for n_iter in t_correct:
        for t in n_iter:
            writer.add_scalar('Accuracy/train', t, o)
            o+=1


    entrer,label,output,test_lose,test_correct = test(testing_data, CNN, loss_fn,device)
    entrer = entrer.to('cpu')
    label = label.to('cpu')
    output = output.to('cpu')

    for n_iter in test_correct:
        writer.add_scalar('Accuracy/test',n_iter, o)
        o+=1

    for n_iter in test_lose:
        writer.add_scalar('Loss/test', n_iter, o)
        o+=1

    writer.close()

    entrer1,label1,output1,a,b = test(testing_data_bruit, CNN, loss_fn,device)
    entrer1 = entrer1.to('cpu')
    label1 = label1.to('cpu')
    output1 = output1.to('cpu')

    entrer2,label2,output2,a,b = test(testing_data_rot, CNN, loss_fn,device)
    entrer2 = entrer2.to('cpu')
    label2 = label2.to('cpu')
    output2 = output2.to('cpu')

    entrer3,label3,output3,a,b = test(testing_data_flou, CNN, loss_fn,device)
    entrer3 = entrer3.to('cpu')
    label3 = label3.to('cpu')
    output3 = output3.to('cpu')
    print("Done!")
    print('Finished Training')

    
    with torch.no_grad():
        number = 20
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(entrer[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(label[index].numpy())
            
            ax = plt.subplot(2, number, index + 1 +number)
            plt.imshow(output[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


        plt.figure(figsize=(20, 4))
        for index in range(number):
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(entrer1[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(str(label1[index].numpy()))
            
            ax = plt.subplot(2, number, index + 1 +number)
            plt.imshow(output1[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

        plt.figure(figsize=(20, 4))
        for index in range(number):
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(entrer2[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(str(label2[index].numpy()))
            
            ax = plt.subplot(2, number, index + 1 +number)
            plt.imshow(output2[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


        plt.figure(figsize=(20, 4))
        for index in range(number):
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(entrer3[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(str(label3[index].numpy()))
            
            ax = plt.subplot(2, number, index + 1 +number)
            plt.imshow(output3[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    # print labels
    # show images
if __name__ == "__main__":
        main()