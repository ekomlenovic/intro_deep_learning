from email.mime import image
from tkinter.font import BOLD
import torch
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary


import Custom as C
import AEV

def mean_std(loader):
  sum, squared_sum, num_batches = 0,0,0
  for data,_ in loader:
    sum += torch.mean(data,dim=[0,2,3])
    squared_sum += torch.mean(data**2,dim=[0,2,3])
    num_batches += 1
  mean = sum/num_batches
  std = (squared_sum/num_batches - mean**2)**0.5
  return mean, std

#tensor([0.5064, 0.4258, 0.3832]) tensor([0.3093, 0.2890, 0.2883])
 
def main():
    trans = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
                                transforms.Resize(128)])
    DataTest = C.CustomImageDataset(annotations_file='../data/img_align_celeba/identity_CelebA.txt'
                                         ,img_dir='../data/img_align_celeba'
                                         ,partition='../data/img_align_celeba/list_eval_partition.txt'
                                         ,train=1,transform = trans)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    tb = SummaryWriter()
    learning_rate = 0.0002
    num_epochs = 100
    images = []
    
    #print(summary(Net(),input_size=(32,3,512,512),dtypes=[torch.float32]))
    loss_fn = nn.MSELoss()

    DataTest = C.LoadData(DataTest,32)
    tb = SummaryWriter()
    
    if input("voulez vous réentrainer le model ? y/n \n") == "n":
        autoencoder = AEV.Net().to(device)
        autoencoder.load_state_dict(torch.load("../data/Model/Visage_AE.pth",device))
        autoencoder.eval()
    else:
        DataTrain = C.CustomImageDataset(annotations_file='../data/img_align_celeba/identity_CelebA.txt',
                        img_dir='../data/img_align_celeba'
                        ,partition='../data/img_align_celeba/list_eval_partition.txt'
                        ,train=0,transform = trans)
        imageViz =  C.CustomImageDataset(annotations_file='../data/img_align_celeba/identity_CelebA.txt',
                        img_dir='../data/img_align_celeba'
                        ,partition='../data/img_align_celeba/list_eval_partition.txt'
                        ,train=2,transform = trans)
        imageViz = np.array(imageViz).reshape(-1,order='F').reshape(2,32)[0].tolist()
        image = torch.cat(imageViz).view(-1,3,156,128).to(device)
        print(image.shape)
        DataTrain = C.LoadData(DataTrain,64)
        autoencoder = AEV.Net().to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(),lr=learning_rate,weight_decay=1e-8)
        print(autoencoder)
        #autoencoder.load_state_dict(torch.load("Model/Visage80.pth",device))
        #autoencoder.eval()
        
        for epoch in range(0,num_epochs+1):
            print(f"Epoch {epoch+1}\n-------------------------------")
            loss = AEV.train(DataTrain, autoencoder, loss_fn, optimizer,device)
            tb.add_scalar("Loss", loss, epoch)
            tb.add_scalar("Average Loss", loss/len(DataTrain)*100, epoch)
            if epoch%5 == 0:
                torch.save(autoencoder.state_dict(), './Model/Visage'+str(epoch)+'.pth')
            
            generated_img = autoencoder(image).view(-1,3,156,128).cpu().detach()
            # make the images as grid
            generated_img = make_grid(generated_img)
            # save the generated torch tensor models to disk
            save_image(generated_img, f"./image_epochAEV/AEV_epoch{epoch}.png")
            images.append(generated_img)
    tb.close()
    X,pred = AEV.test(DataTest, autoencoder, loss_fn,device)
    X = X.to("cpu")
    pred = pred.to("cpu")
    
    
    transform=transforms.Compose([transforms.ConvertImageDtype(torch.float64),transforms.ToPILImage()])
    imgs = [np.array(transform(img)) for img in images]
    imageio.mimsave('./image_epochAEV/AEV_images.gif', imgs)

    
    print("Done!")
    print('Finished Training')
    
    with torch.no_grad():
        number = 20
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            
            image = transform(torch.from_numpy(X[index].numpy().reshape(3,156,128)))
            imagepred = transform(torch.from_numpy(pred[index].numpy().reshape(3,156,128)))
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(image)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, number, index + 1 +number)
            
            plt.imshow(imagepred)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
    





if __name__ == "__main__":
        main()