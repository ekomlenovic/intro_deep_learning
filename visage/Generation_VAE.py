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
import VAE_Visage as VAE

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

import sys


def main():
        transform=transforms.Compose([transforms.ConvertImageDtype(torch.float64),transforms.ToPILImage()])
                    
        trans = transforms.Compose([transforms.ConvertImageDtype(torch.float32),transforms.Resize(128)])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Gen = VAE.VAE().to(device)
        Gen.load_state_dict(torch.load("./data/Model/Visage_VAE.pth",device))
        Gen.eval()
        
        
        
        
        with torch.no_grad():
            for i in range(0,11):
                z = Gen.reparameterize(torch.randn(512).to(device),torch.randn(512).to(device))
                generated = Gen.decoder(z)
                generated = transform(torch.from_numpy(generated.to("cpu").numpy().reshape(3,156,128)))
                ax = plt.subplot(1, 11, i + 1)
                plt.imshow(generated)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()
        
if __name__ == "__main__":
        main()