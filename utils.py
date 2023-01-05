import torch
import torch.nn
import numpy as np
import math
import numpy as np
import torch
import json
import imageio as iio
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw 
import random
import torch.nn.functional as F
from Network import *
from dataset import *
chunksize = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeRF().to(device)

def camera2world_coords(pose):
    '''
    imput : Transformation matrix
    Returns: x,y,z and Rotation in world coordinates 
    '''
    return pose[:-1,-1], pose[:-1,:-1]


def get_ray_bundle(height, width, focal_length, pose):

    world_x,rot = camera2world_coords(pose)
    i, j = torch.meshgrid(torch.arange(height).to(pose),torch.arange(width).to(pose))
    directions = torch.stack([(j - width * 0.5) / focal_length,-(i - height * 0.5) / focal_length,-torch.ones_like(i)], dim=-1)
    # add a dimension in the last two dimention (h,w,1,3)
    # to perform matrix mul vector: (h,w,1,3) * (3,3) = (h,w,3,3)
    # then sum in the last dimension: sum((h,w,3,3),-1) = (h,w,3)
    ray_directions = torch.sum(directions[..., None, :] * rot, dim=-1)
    # ray origin is just cam2world[:3, -1]
    ray_origins = world_x.expand(ray_directions.shape)
    return ray_origins, ray_directions



def positional_encoding(x,f=6):
    '''input : x is a tensor, f values are embedding to x in function'''
    out= [x]
    func = [torch.sin, torch.cos]
    for i in range(0,f):
        for f in func:
            out.append(f(2**i * x))
    return torch.concat(out,-1)


def stratified_sampling(rays_o,rays_d,nearest = 2,farthest=6,num_samples = 64, method = 'inverse_depth', randomize= False):
    '''input : 5D input, nearest point on a ray, farthest, number of samples to draw
       Divide the ray to different strata os subsections (continous), draw samples from each
       different methods, either sample linearly between nearest and farthest point or
       linearly in inverse depth '''
    
    strata = torch.linspace(nearest,farthest,num_samples).to(rays_o) 
    if method == 'inverse_depth':
        samples = 1/(1/nearest*(1-strata) +1/farthest*(strata))
    elif method == 'near2far':
        samples = torch.linspace(nearest,farthest,num_samples) 

    if randomize:
        alpha = 0.1
        gamma = 0.003  # parameter that controls degree of randomness
        mean = torch.mean(samples)
        for i in range(0,int(alpha*num_samples)):
            rand_idx = random.randint(0,num_samples-1)
            samples[rand_idx] += random.randint(0,num_samples)*gamma*(samples[rand_idx+1]-samples[rand_idx-1])

    samples = samples.expand(rays_o.shape[0],rays_o.shape[1],num_samples) # h x w x num_samples
    # now apply transform (scale with view directions and offset by position)
    # to get sampled points along the ray of an image
    ray_points = rays_o[:,:, np.newaxis, :] + rays_d[:,:, np.newaxis, :] * samples[:,:, :, np.newaxis]

    return samples,ray_points 





def camera2world_coords(pose):
    '''
    imput : Transformation matrix
    Returns: x,y,z and Rotation in world coordinates 
    '''
    return pose[:-1,-1], pose[:-1,:-1]


def get_rays(height, width, focal_length, pose):
    world_x,rot = camera2world_coords(pose)
    i, j = torch.meshgrid(torch.arange(height).to(pose),torch.arange(width).to(pose))
    directions = torch.stack([(j - width * 0.5) / focal_length,-(i - height * 0.5) / focal_length,-torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(directions[..., None, :] * rot, dim=-1)
    rays_o = world_x.expand(rays_d.shape)
    return rays_o, rays_d



def stratified_sampling(rays_o, rays_d, nearest, farthest, num_samples, randomize = True):
  
  depth_values = torch.linspace(nearest, farthest, num_samples).to(rays_o) # zvalues
  if randomize:
    noise_shape = list(rays_o.shape[:-1]) + [num_samples] # noise remove weird artifacts in rendering
    depth_values = depth_values + torch.rand(noise_shape).to(rays_o) * (farthest- nearest) / num_samples
  z_values = rays_o[..., None, :] + rays_d[..., None, :] * depth_values[..., :, None]
  length = torch.sqrt(torch.sum(rays_d * rays_d, dim = -1))
  rays_d = rays_d / length[..., None]
  query_dirs = rays_d[..., None, :].expand(z_values.shape)
  return z_values, depth_values, query_dirs



def volume_rendering(radiance_field, ray_origins, depth_values):
  #compute each query point
  sigma_a = F.relu(radiance_field[..., 3])
  rgb = torch.sigmoid(radiance_field[..., :3])
  one_e_10 = torch.tensor([1e10]).to(ray_origins)
  dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
  alpha = 1. - torch.exp(-sigma_a * dists)
  # cumulative product (one at the begining)
  cumprod = torch.cumprod(1.-alpha+ 1e-10, -1)
  cumprod = torch.roll(cumprod, 1, -1)
  cumprod[..., 0] = 1.
  cumprod = torch.Tensor(cumprod)
  weights = alpha * cumprod
  rgb_map = (weights[..., None] * rgb).sum(dim=-2)
  return rgb_map


def positional_encoding(x,f=6):
    '''input : x is a tensor, L values are embedding to x in function'''
    out= [x]
    func = [torch.sin, torch.cos]
    for i in range(0,f):
        for f in func:
            out.append(f(2**i * x))
    return torch.concat(out,-1)


def batchify(inputs: torch.Tensor, chunksize = 1024):
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def render_rays(height, width, focal_length, pose, nearest, farthest, num_samples, num_encoding, encoding_function, batchify):
  
  rays_o, rays_d = get_rays(height, width, focal_length, pose)
  # Sample query points along each ray (each pixels)
  z_values, depth_values, query_directions = stratified_sampling(rays_o, rays_d, nearest, farthest, num_samples)
  # flatten zvalues
  flat_points = z_values.reshape((-1, 3))
  flat_directions = query_directions.reshape((-1, 3))
  encoded_points = encoding_function(flat_points)
  encoded_dirs = encoding_function(flat_directions)
  encoded_inputs = torch.cat([encoded_points, encoded_dirs], dim = -1)

  # Split the encoded points into batches
  batches = batchify(encoded_inputs, chunksize=chunksize)

  predictions = []
  half = 3 + 6 * num_encoding
  for batch in batches:
    predictions.append(model(batch[:,:half], batch[:,half:]))
  radiance_field_flattened = torch.cat(predictions, dim=0)

  unflattened_shape = list(z_values.shape[:-1]) + [4]
  radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

  # Perform differentiable volume rendering to get back the RGB image.
  rgb_predicted = volume_rendering(radiance_field, rays_o, depth_values)
  return rgb_predicted

