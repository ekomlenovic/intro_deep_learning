from email.mime import image
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
import PIL
from PIL import Image


from torchinfo import summary


import Custom as C
import AEV



def main():
    trans = transforms.Compose([transforms.ConvertImageDtype(torch.float32),transforms.Resize(128)])
    DataTest = C.CustomImageDataset(annotations_file='../data/img_align_celeba/identity_CelebA.txt'
                                         ,img_dir='../data/img_align_celeba'
                                         ,partition='../data/img_align_celeba/list_eval_partition.txt'
                                         ,train=2,transform = trans)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    learning_rate = 0.001
    num_epochs = 20
    
    
    #print(summary(Net(),input_size=(32,3,512,512),dtypes=[torch.float32]))
    loss_fn = nn.MSELoss()
    CNN = AEV.Net().to(device)
    DataTest = C.LoadData(DataTest,32)
    plt.figure(figsize=(20, 20))
    number = 16
    for i in range(10):
        CNN.load_state_dict(torch.load("../data/Model/Visage"+str(i*10)+".pth",device))
        CNN.eval()
        X,pred = AEV.test(DataTest, CNN, loss_fn,device)
        X = X.to("cpu")
        pred = pred.to("cpu")

        torch.cuda.empty_cache()
        
        with torch.no_grad():
            
            
            for index in range(number):
                # display original
                transform=transforms.Compose([transforms.ConvertImageDtype(torch.float64),transforms.ToPILImage()])
                image = transform(torch.from_numpy(X[index].numpy().reshape(3,156,128)))
                imagepred = transform(torch.from_numpy(pred[index].numpy().reshape(3,156,128)))
                imagepred.save(f"imagecree/{index+(i * number+1)}.png", format="png")
                ax = plt.subplot(11, number, index + 1)
                plt.imshow(image)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax = plt.subplot(11, number, index+1 +number*(i+1))
                plt.imshow(imagepred)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
    plt.show()
    
if __name__ == "__main__":
        main()