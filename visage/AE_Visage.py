import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1)  
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)
        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,256,3,1)
        self.conv8 = nn.Conv2d(256,256,3,1)
        self.conv9 = nn.Conv2d(256,512,3,1)
        self.Dropout1 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(23040 ,2**12)
        self.fc3 = nn.Linear(2**12,2**11)
        self.fc4 = nn.Linear(2**11,2**10)
        self.fc5 = nn.Linear(2**10,2**9)
        
        self.decoder2= nn.Linear(2**9,2**10)
        self.decoder3 = nn.Linear(2**10,2**11)
        self.decoder4 = nn.Linear(2**11,156*128*3)
        
    def encode(self, x,flatt=1):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.Dropout1(x)
        x = torch.flatten(x,flatt)
        x = F.relu(self.fc1(x))
        x = self.Dropout1(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def decode(self, x):

        x = F.relu(self.decoder2(x))
        x = F.relu(self.decoder3(x))
        x = torch.sigmoid(self.decoder4(x))
        return x

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)

        return x
    
def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset) 
    total_loss = 0#calcule la taille du jeu de donnée d'entrainement
    for batch, (X, _) in enumerate(dataloader):   #donne l'index de l'image X=tensor de l'image de taille 28*28 de la taille du bash chargé       
        X= X.to(device) #On charge les tensors dans le gpu ou dans le cpu
        # Compute prediction error
        pred = model(X) #On prédit y
        X=X.reshape(-1, 156*128*3)
        loss = loss_fn(pred, X) #On regarde le loss
        total_loss += loss.item()


        # Backpropagation optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 32 == 0: 
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  {(current/size*100):>2f}%")
        
        del X
        torch.cuda.empty_cache()
    return total_loss

def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X , _ in dataloader:
            X= X.to(device)
            pred = model(X)
            X=X.reshape(-1, 156*128*3)
            test_loss += loss_fn(pred, X).item()
            
            torch.cuda.empty_cache()
        
    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return X,pred