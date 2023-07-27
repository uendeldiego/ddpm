import torch
import torchvision
import matplotlib.pyplot as plt

# Define a função que vai plotar algumas imagens aleatórias do dataset escolhido
def show_images(datset, num_samples=4, cols=4):

# Plota algumas amostras de imagens do dataset
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])

data = torchvision.datasets.FashionMNIST('/home/diego/Documentos/ddpm/data', download=False)
# data = "/home/diego/Documentos/ddpm/data/"
show_images(data)