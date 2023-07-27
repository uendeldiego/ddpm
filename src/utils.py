import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import math
from torch import nn
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from foward_diffusion import get_index_from_list



def show_tensor_image(image, i):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imsave('../logs/show_tensor_'+str(i)+'.png', reverse_transforms(image))#plt.imshow(reverse_transforms(image))

@torch.no_grad()
def sample_timestep(x, t, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, model, posterior_variance):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(IMG_SIZE=64, device='cpu', T=None, sqrt_one_minus_alphas_cumprod=None, betas=None, sqrt_recip_alphas=None, model=None, posterior_variance=None):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    #plt.figure(figsize=(25,25))
    #plt.axis('off')
    num_images = 10
    stepsize = int(T/int(num_images*.5))

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, model, posterior_variance)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            #plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu(), i)
    #plt.show()      