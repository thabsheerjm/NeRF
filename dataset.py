import math
import numpy as np
import torch
import json
import imageio as iio
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dir ='./dataset/lego/transforms_train.json'  
test_dir ='./dataset/lego/transforms_test.json' 
val_dir ='./dataset/lego/transforms_val.json'
collection = [train_dir,test_dir,val_dir] 

imgs_path = './dataset/lego'


data = []
for i in collection:
    with open(i) as file:
        data.append(json.load(file))



def extract_data(dataset):
    frames = dataset["frames"]
    camera_angle_x = dataset["camera_angle_x"]
    imgs = []
    rotation = []
    poses = []
    for data in frames:
        img_dir = imgs_path+data['file_path'].replace('.','')+'.png' #remove the dot in between
        rotation.append(data['rotation'])
        poses.append(np.array(data['transform_matrix'],dtype = np.float32))
        img = Image.open(img_dir)
        # img = np.array(img)
        img = img.convert('RGB').resize((100,100))
        img = np.array(img,dtype=np.float32)
        img/=255
        # img = Image.fromarray(img).resize((size,size))
        # plt.imshow(img)
        # plt.show()
        imgs.append(img)  #list of imagesprint(H)
    # height, width = imgs[0].shape
    height = img.shape[1]
    width = img.shape[0]
    focal = 0.5 *width / np.tan(0.5 * camera_angle_x)
    return np.array(imgs,dtype=np.float32), np.array(poses),rotation, (height,width), np.array(focal)
        

images, poses,rotation, (height,width), focal_length = extract_data(data[0])
vimages, vposes,vrotation, (vheight,vwidth), vfocal_length = extract_data(data[2])  
test_image, test_pose = vimages[10], vposes[10]

# Move data to the device ( Gpu : if Gpu )
images = torch.from_numpy(images[:100, ..., :3]).to(device)
poses = torch.from_numpy(poses).to(device)
focal_length = torch.from_numpy(focal_length).to(device)
test_image = torch.from_numpy(test_image).to(device)
test_pose = torch.from_numpy(test_pose).to(device)

# Show test img
# plt.imshow(test_img.detach().cpu().numpy())
# plt.show()