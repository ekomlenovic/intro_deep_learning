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
import VAE



def ShowImage(Data):
    for _, (x,y) in enumerate(Data):
        plt.figure(figsize=(16,4))
        for index in range(32):
            # display original
            transform=transforms.Compose([transforms.ConvertImageDtype(torch.float64),transforms.ToPILImage()])
            image = transform(x[index])
            ax = plt.subplot(2, 16, index + 1)
            plt.title(y[index].numpy())
            plt.imshow(image)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        break
    plt.show()
 
 

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
    if input("voulez vous réentrainer le model ? y/n \n ") == "n":
        autoencoder_var = VAE.VAE().to(device)
        autoencoder_var.load_state_dict(torch.load("../data/Model/Visage_VAE.pth",device))
        autoencoder_var.eval()
    else:
        DataTrain = C.CustomImageDataset(annotations_file='../data/img_align_celeba/identity_CelebA.txt',
                        img_dir='../data/img_align_celeba'
                        ,partition='../data/img_align_celeba/list_eval_partition.txt'
                        ,train=0,transform = trans)
        DataTrain = C.LoadData(DataTrain,64)
        
        imageViz =  C.CustomImageDataset(annotations_file='../data/img_align_celeba/identity_CelebA.txt',
                        img_dir='../data/img_align_celeba'
                        ,partition='../data/img_align_celeba/list_eval_partition.txt'
                        ,train=2,transform = trans)
        imageViz = np.array(imageViz).reshape(-1,order='F').reshape(2,32)[0].tolist()
        image = torch.cat(imageViz).view(-1,3,156,128).to(device)
        
        autoencoder_var = VAE.VAE().to(device)
        optimizer = torch.optim.Adam(autoencoder_var.parameters(),lr=learning_rate,weight_decay=1e-8,betas=(0.5,0.999))
        #autoencoder_var.load_state_dict(torch.load("Model/VisageVAE.pth",device))
        #autoencoder_var.eval()
        print(autoencoder_var)
        
        for epoch in range(0,num_epochs+1):
            print(f"Epoch {epoch+1}\n-------------------------------")
            loss = VAE.train(DataTrain, autoencoder_var, optimizer,device)
            tb.add_scalar("Loss", loss, epoch)
            tb.add_scalar("Average Loss", loss/len(DataTrain)*100, epoch)
            
            generated_img,_,_ = autoencoder_var(image)
            generated_img = generated_img.view(-1,3,156,128).cpu().detach()
            # make the images as grid
            generated_img = make_grid(generated_img)
            # save the generated torch tensor models to disk
            save_image(generated_img, f"./image_epochVAE/VAE_epoch{epoch}.png")
            images.append(generated_img)
        
        torch.save(autoencoder_var.state_dict(), '../data/Model/Visage_VAE.pth')
        
    tb.close()
    X,pred = VAE.test(DataTest, autoencoder_var, loss_fn,device)
    X = X.to("cpu")
    pred = pred.to("cpu")
    
    transform=transforms.Compose([transforms.ConvertImageDtype(torch.float64),transforms.ToPILImage()])
    imgs= [np.array(transform(img)) for img in images]
    imageio.mimsave('./image_epochVAE/VAE_images.gif', imgs)
    
    print("Done!")
    print('Finished Training')
    
    with torch.no_grad():
        number = 20
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            transform=transforms.Compose([transforms.ConvertImageDtype(torch.float64),transforms.ToPILImage()])
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