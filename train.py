import numpy as np
import torch
import torch.nn.functional as F

from Network import NeRF
from utils import *
from dataset import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
num_encoding = 6
nearest = 2.0
farthest = 6.0
num_samples =  64
chunksize = 1024
encoder = lambda x: positional_encoding(x)

# learning parameters
lr = 2e-3  #5e-3
num_iters = 100_000

# record at intervals 
interval = 1000

iterations = []

# define the model
model = NeRF()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# start training
for i in range(num_iters):
   
  # Randomly pick an image as the target.
  target_img_idx = np.random.randint(images.shape[0])
  target_img = images[target_img_idx].to(device)
  target_pose = poses[target_img_idx].to(device)
  
  predicted_img = render_rays(height, width, focal_length,target_pose, nearest,farthest, num_samples, num_encoding, encoder, batchify)
  

  loss = F.mse_loss(predicted_img, target_img)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  target_img.to(device)
  target_pose.to(device)
  # Record progress
  if i % interval == 0:
    
    # Render the held-out view
    predicted_img = render_rays(height, width, focal_length,test_pose, nearest,farthest, num_samples,num_encoding,encoder, batchify)
    loss = F.mse_loss(predicted_img, test_image)
    print("Loss:", loss.item())
    iterations.append(i)

    # plt.figure(figsize=(10, 4))
    # plt.imshow(predicted_img.detach().cpu().numpy())
    # plt.title(f"Iteration {i}")
    # plt.show()

print('Done training! , savinf the model ....')
torch.save(model.state_dict(),'./   NerfModel_params')
torch.save(model,'./Nerf_model.pth')
print('Model Saved')