from dataset_stanford_cars import load_data_cars, load_transformed_dataset
from model import get_loss,  SimpleUnet
from utils import sample_plot_image
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.optim import Adam
from foward_diffusion import forward_data

IMG_SIZE = 64
BATCH_SIZE = 16

#data = load_data_cars()

data = load_transformed_dataset(IMG_SIZE)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


model = SimpleUnet()
# print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.00025)
epochs = 200 # Try more!

#pytorch_cuda_alloc_config(0.6, max_split_size_mb, 128)
#pytorch_cuda_alloc_config = garbage_collection_threshold(0.6, max_split_size_mb, 128)

_forward_data = forward_data()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, _forward_data["T"], (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if epoch % 1 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image(IMG_SIZE=IMG_SIZE, device=device, T=_forward_data["T"], sqrt_one_minus_alphas_cumprod=_forward_data["sqrt_one_minus_alphas_cumprod"], betas = _forward_data["betas"], sqrt_recip_alphas = _forward_data["sqrt_recip_alphas"], model = model,posterior_variance = _forward_data["posterior_variance"] ) # IMG_SIZE=64, device='cpu', T=None, sqrt_one_minus_alphas_cumprod=None, betas=None, sqrt_recip_alphas=None, model=None, posterior_variance=None):

# train loop



# Save model -> models

#torch.save(model, 'models/modelo1.pt')
