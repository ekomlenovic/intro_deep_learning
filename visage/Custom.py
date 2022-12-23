import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from torch.utils.data import DataLoader
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file,img_dir,partition=None,  transform=None, target_transform=None,train = 1):
        self.img_labels = pd.read_csv(annotations_file,sep=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.partition = pd.read_csv(partition,sep= " ")
        self.imgTrain = pd.merge(self.img_labels,self.partition).loc[self.partition['partition'] == 0]
        self.imgTest = pd.merge(self.img_labels,self.partition).loc[(self.partition['partition'] == 1) | (self.partition['partition'] == 2) ]
        self.imgViz = pd.merge(self.img_labels,self.partition).loc[self.img_labels.index.isin(range(202599-32,202599+1,1))]
        self.ImgInt = pd.merge(self.img_labels,self.partition).loc[self.img_labels.index.isin(np.random.random_integers(1,202599,size = 2))]
        self.train = train
        if self.train == 0:
            self.data = self.imgTrain
        elif self.train == 1:
            self.data = self.imgTest
        elif self.train == 2:
            self.data = self.imgViz
        else :
             self.data = self.ImgInt


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.data.iloc[idx,0])
        image = read_image(img_path)
        label = self.data.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def LoadData(Datatrain,batch_size):
    

    training_data_load = DataLoader(Datatrain,batch_size=batch_size,shuffle = True)
    
    return training_data_load
