from dataset_stanford_cars import load_data_cars, load_transformed_dataset
from model import cosine_beta_schedule, get_loss, sample_plot_image,  SimpleUnet
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch




data = load_data_cars()






# Define o tamanho das imagens e o tamanho do batch
# Por limitações de hardware, este código rodando no notebook Avell do PAVIC não 
# permite rodar com tamanhos de imagens maiores 256

IMG_SIZE = 64
BATCH_SIZE = 16

data = load_transformed_dataset(IMG_SIZE)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)



from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.00025)
epochs = 200 # Try more!

#pytorch_cuda_alloc_config(0.6, max_split_size_mb, 128)
#pytorch_cuda_alloc_config = garbage_collection_threshold(0.6, max_split_size_mb, 128)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if epoch % 1 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image()

# train loop



# Save model -> models

#torch.save(model, 'models/modelo1.pt')
