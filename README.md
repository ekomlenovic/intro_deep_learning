# projet Deep Learning

**Visage** réalisé en Python dans le cadre de l'UE LIFPROJET de [l'Université Lyon 1 Claude Bernard](http://www.univ-lyon1.fr/) durant l'année 2022-2023. 

## Auteurs

* Louis-Antoine Pham 
* Alexandre Faure 
* Emilien Komlenovic 

## I. Installation

### Anaconda
* Installer [Anaconda](https://www.anaconda.com/products/distribution) puis créer un environnement.
    1. conda create -n p39 python=3.9
    2. activate p39
* Installer [Pytorch](https://pytorch.org/get-started/locally/).
    1. Avec [Cuda](https://developer.nvidia.com/cuda-zone)  
    `$ conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`
    2. Sans Cuda (Seulement CPU [Non conseillé])  
    `$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`
* Installer matplotlib.  
    `$ conda install matplotlib`
* Dans l'environnement p39.  
    `$ pip install torchinfo opencv-python tk pandas tensorboard`
* Télécharger aussi la base donnée [CelebA](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) puis l'extraire dans data.  
* Télécharger aussi les [modeles](https://drive.google.com/drive/folders/1BUj8onGfuyus-86VYSmgT8tyfARI4f-c?usp=sharing) puis les placer dans /visage/Model.


## II. Structure des fichiers

* Le Dossier **Data** contient 2 sous dossiers :
    1. **MNIST** qui fu notre base de test d'initiation.
    2. **img_align_celeba** qui est la base de donnée utilisée pour le projet.
* Le Dossier **learn** qui contient 2 sous dossiers:
    1. **lean_basis** qui contient tout nos entrainement sur la base MNSIT ainsi que sur un nuage de points.
    2. **Model** qui est la sauvegarde de nos modèles d'entrainement.
* Le Dossier **visage** qui contient le code principale de notre projet.  

## III. Lancement de l'Application de Démonstration  

L'application nommé "App.py" se situe à la racine. Elle présente une génération de visage avec un Auto-Encoder, Un Auto-Encoder Variationnel, ainsi qu'une interpolation entre deux visages avec les technologies cité précédemment.