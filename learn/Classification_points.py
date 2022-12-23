from __future__ import absolute_import

from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo as info


from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

#writer = SummaryWriter('runs/Classisication_experiment_1')


#classe de réseau de neurone de 4 couches 
#1ère couche:2 neurones
#2ère couche:64 neurones
#3ère couche:64 neurones
#4ère couche:2 neurones
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       
       self.fc1 = nn.Linear(2, 2**7)
       self.fc2 = nn.Linear(2**7, 2**7)
       self.fc3 = nn.Linear(2**7, 3)
       
       
       #chance de dropout de 10%
       self.dropout = nn.Dropout(0.5)
   
   #définition des fonction d'action à chaques couches 
   def forward(self, x):
        #fonction d'activation
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



# Generation of one point (one sample)
def one_sample():
    #crée une liste x de 2 valeurs aléatoires
    x = np.array( [ 3.141592*4*np.random.ranf(), 2.0*np.random.ranf()-1 ])
    
    if (x[1]<0.25 and x[1]>-0.25):
        y = np.array([0,0,1])
    else:  
        if (np.cos(x[0]) < x[1]):
            y = np.array([ 0, 1,0])
        else:
            #[1,0] si x[1] est au-dessus de cosinus
            y = np.array([1, 0,0])
        
    return x,y


# Generation of a batch of points (batch of samples)
#créer un tableau de 2 listes x une liste de coordonées et y les valeurs

def next_batch(n):
    
    x = np.zeros( shape=(n,2), dtype=np.float32)
    y = np.zeros( shape=(n,3), dtype=np.int32)
    for i in range(0, n):
        x[i],y[i] = one_sample()
    return x,y




def main():
    
    
    ############# NETWORK definition/configuration
    #création d'un réseau
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    net = Net()
    net.to(torch.device('cuda'))
    print(net)
    ############# Stochastic Gradient Descent Config
    criterion = nn.CrossEntropyLoss()
    #optimisation par descente de gradiant
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

    ############ TRAINNING
    x_train, y_train = next_batch(1000)
    x_train = torch.from_numpy(x_train).to(torch.device('cuda'))
    y_train = torch.from_numpy(y_train).long().to(torch.device('cuda'))
    for epoch in range(15):

# loop over the dataset multiple times
        running_loss = 0.0
        for i in range(1000):         # iterate on mini batches. mani-batch = a subset of the database

         
            # zero the parameter gradients

            optimizer.zero_grad()
        
            # forward + backward + optimize
        
            y_pred = net(x_train)

    
            loss = criterion(y_pred, torch.max(y_train, 1)[1] )
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:   
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,  running_loss/ 100))
                #writer.add_scalar('Loss/train', running_loss/100, epoch*1000+i)
                running_loss = 0.0    	

    ########### Drawing of the point clound with good or bad classification
    torch.save(net.state_dict(), '../data/Model_learn/Classification_points.pth')
    plt.figure(1)
    #création de l'echantillon de test de test
    x_test, y_test = next_batch(10000)
    x_test = torch.from_numpy(x_test).to(torch.device('cuda'))
    y_test = torch.from_numpy(y_test).long().to(torch.device('cuda'))
    
    y_pred = net(x_test)

    y_pred = y_pred.detach().to('cpu').numpy()
    x_test = x_test.detach().to('cpu').numpy()
    y_test = y_test.detach().to('cpu').numpy()
    error = 0
    for i in range(len(y_test)):
        s = y_test[i]
        if ( np.argmax(s)!= np.argmax(y_pred[i]) ):
            error += 1
            plt.plot( x_test[i,0], x_test[i,1], 'ro', color='red',markersize = 5)
        else:
            if (np.argmax(s)==1):
                plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='green',markersize = 5)
            elif (np.argmax(s)==2):
                plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='black',markersize = 5)
            else:
                plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='blue',markersize = 5)

    
    plt.show()
    print("nb d'erreur : " + str(error/len(y_test)))
    
    info.summary(net,input_size=(1,1,2))

if __name__ == "__main__":
        main()