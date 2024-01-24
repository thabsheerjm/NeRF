from PIL import Image
import numpy as np
import glob


def np2img(imgs):
    count = 0
    for data in imgs:
        count+=1
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

        im = Image.fromarray(rescaled)
        im.save('./predicted_images/test'+str(count)+'.png')



def make_gif(frame_folder):
    file_name = ['./predicted_images/test'+str(i)+'.png' for i in range(1,201)]
    frames = [Image.open(image) for image in file_name]

    frame_one = frames[0]
    frame_one.save("Nerf.gif", format="GIF", append_images=frames,
               save_all=True, duration=10, loop=1)




if __name__=="__main__":
    images = np.load('Predicted_images.npy')
    np2img(images)
    make_gif("./predicted_images")