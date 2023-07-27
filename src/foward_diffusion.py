import torch
import cv2
from dataset_stanford_cars import load_transformed_dataset
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
import torch.nn.functional as F
import math
from torch import nn
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
# Define o beta schedule (agendador de passos/slices)
# Ele pode ser de várias formas. Aqui temos um beta schedule linear e um cossenóide

# Beta linear
def linear_beta_schedule(timesteps, start=0.001, end=0.02):
    return torch.linspace(start, end, timesteps)

# Beta cossenóide
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# Define a função que vai retornar índice t específico de uma lista passada de 
# valores "vals" enquanto considera a dimensão do batch (lote)
def forward_data():
    T = 400
    betas = cosine_beta_schedule(timesteps=T)

    # Pré calcular os diferentes termos para a forma fechada da difusão
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return {
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }
T = 400
betas = cosine_beta_schedule(timesteps=T)

# Pré calcular os diferentes termos para a forma fechada da difusão
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):

# Retorna um índice t específico de uma lista passada de valores vals enquanto considera a dimensão do batch (lote)
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
 
    
# Define a função que vai gerar amostras da etapa da difusão direta
def forward_diffusion_sample(x_0, t, device="cpu", sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod):

# Toma uma imagem e timestep (um intervalo de tempo) como entrada e retorna a versão desta com ruído

    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
# Média + variância (mean + variance)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)