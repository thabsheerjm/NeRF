import torch
import numpy as np

# from train import encoder
from Network import NeRF
from utils import *
from dataset import *
# load the model
model = NeRF()
model = torch.load('./Nerf_model.pth')

encoder = lambda x: positional_encoding(x)


images, poses,rotation, (height,width), focal_length = extract_data(data[1]) 
focal_length = torch.from_numpy(focal_length).to(device)
im_list = []
for i in poses:
    i = torch.Tensor(i).to(device)
    predicted_img = render_rays(height, width, focal_length, i,2, 6,64, 6,encoder,batchify)
    im_list.append(predicted_img.detach().cpu().numpy())

# saqving the rendered test images
np.save('Predicted_images',im_list)
imgs = np.load('Predicted_images.npy')
plt.imshow(imgs[21])
plt.show()