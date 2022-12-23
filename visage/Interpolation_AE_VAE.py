from email.mime import image
from re import I
import torch
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary


import Custom as C
import AE_Visage as AEV
import VAE_Visage as VAE

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

import sys


def main(*args):
    transform=transforms.Compose([transforms.ConvertImageDtype(torch.float64),transforms.ToPILImage()])
                            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Chargement AutoEncoder
    AE = AEV.Net().to(device)
    AE.load_state_dict(torch.load("./data/Model/Visage_AE.pth",device))
    AE.eval()

    #Chargement AutoEncoder Variationel
    VAE_Interpolation = VAE.VAE().to(device)
    VAE_Interpolation.load_state_dict(torch.load("./data/Model/Visage_VAE.pth",device))
    VAE_Interpolation.eval()
    ax = plt.subplot(1, 13,1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    try:
        if args[1]:
                print("Interpolation entre ",args[1]," et ",args[2])
                trans = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32),transforms.Resize((156,128))])
        
                image1 = Image.open(args[1]).resize((250, 250), Image.Resampling.LANCZOS)
                plt.imshow(image1)
                image1 = trans(image1)
                image2 = Image.open(args[2]).resize((250, 250), Image.Resampling.LANCZOS)
                ax = plt.subplot(1, 13,13)
                plt.imshow(image2)


                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                image2 = trans(image2)

                im1_AE = AE.encode(image1.to(device),0)                        
                im2_AE = AE.encode(image2.to(device),0)
                        
                im1_VAE_mu, im1_VAE_logVar = VAE_Interpolation.encoder(image1.to(device),0)
                im2_VAE_mu, im2_VAE_logVar = VAE_Interpolation.encoder(image2.to(device),0)

                with torch.no_grad():
                    for i in range(0,11):
                        img12_AE = (im1_AE*(1-i/10) + im2_AE*(i/10))
                        img12_AE = AE.decode(img12_AE)
                        imagepred_AE = transform(torch.from_numpy(img12_AE.to("cpu").numpy().reshape(3,156,128)))


                        img12_VAE= VAE_Interpolation.reparameterize(im1_VAE_mu*(1-i/10) + im2_VAE_mu*(i/10), im1_VAE_logVar*(1-i/10) + im2_VAE_logVar*(i/10))
                        img12_VAE = VAE_Interpolation.decoder(img12_VAE)
                        imagepred_VAE = transform(torch.from_numpy(img12_VAE.to("cpu").numpy().reshape(3,156,128)))
                        
                        ax = plt.subplot(2, 13, i + 2)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        plt.imshow(imagepred_AE)
                        ax = plt.subplot(2, 13, i + 2 + 13)
                        plt.imshow(imagepred_VAE)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)

                            
                    plt.show()
    except:
        print("Interpolation Entre Deux Images Aléatoires de CelebA")

        trans = transforms.Compose([transforms.ConvertImageDtype(torch.float32),transforms.Resize(128)])
        ImInter = C.CustomImageDataset(annotations_file='./data/img_align_celeba/identity_CelebA.txt'
                                                ,img_dir='./data/img_align_celeba'
                                                ,partition='./data/img_align_celeba/list_eval_partition.txt'
                                                ,train=4,transform = trans)

    

        ax = plt.subplot(1, 13,1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        image1 = transform(ImInter[0][0])
        image2 = transform(ImInter[1][0])
        im1_AE = AE.encode(ImInter[0][0].to(device),0)
        im2_AE = AE.encode(ImInter[1][0].to(device),0)

        
        im1_VAE_mu, im1_VAE_logVar = VAE_Interpolation.encoder(ImInter[0][0].to(device),0)
        im2_VAE_mu, im2_VAE_logVar = VAE_Interpolation.encoder(ImInter[1][0].to(device),0)

        plt.imshow(image1)
        with torch.no_grad():
            for i in range(0,11):
                img12_AE = (im1_AE*(1-i/10) + im2_AE*(i/10))
                img12_AE = AE.decode(img12_AE)
                imagepred_AE = transform(torch.from_numpy(img12_AE.to("cpu").numpy().reshape(3,156,128)))

                img12_VAE= VAE_Interpolation.reparameterize(im1_VAE_mu*(1-i/10) + im2_VAE_mu*(i/10), im1_VAE_logVar*(1-i/10) + im2_VAE_logVar*(i/10))
                img12_VAE = VAE_Interpolation.decoder(img12_VAE)
                imagepred_VAE = transform(torch.from_numpy(img12_VAE.to("cpu").numpy().reshape(3,156,128)))
                
                ax = plt.subplot(2, 13, i + 2)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.imshow(imagepred_AE)
                ax = plt.subplot(2, 13, i + 2 + 13)
                plt.imshow(imagepred_VAE)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            ax = plt.subplot(1, 13,13)
            plt.imshow(image2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        
if __name__ == "__main__":
        main(*sys.argv)