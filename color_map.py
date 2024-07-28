# ml
import numpy as np

# import PIL
import PIL

# plotting
import matplotlib.pyplot as plt

# misc
import os
import math
import random
import argparse
from tqdm import tqdm

# calculate the mean squared error
def mse(gt, render):
    try:
        return ((gt - render) ** 2)
    except:
        gt = np.transpose(gt)
        return ((gt - render) ** 2)


# calculate the mean absolute error
def mae(gt, render):
    # print(gt,render)
    try:
        return (np.abs(gt - render))
    except:
        gt = np.transpose(gt)
        return (np.abs(gt - render))
    

def get_color_map(gt_path, render_path, save_path, mse_enabled, mae_enabled):
    tmp = 2
    print(f"{tmp:3d}")
    # get all files in the directories
    gt_files = sorted(os.listdir(gt_path))
    render_files = sorted(os.listdir(render_path))
    print(gt_files, render_files)
    
    print("loading images: ")
    gt_images = [np.array(PIL.Image.open(os.path.join(gt_path, f))) for f in tqdm(gt_files)]
    render_images = [np.array(PIL.Image.open(os.path.join(render_path, f))) for f in tqdm(gt_files)]

    # TODO: add index check

    # # convert to numpy arrays
    # gt_images = [np.array(img) for img in gt_images]
    # render_images = [np.array(img) for img in render_images]    

    #gray scale the images
    print("gray scaling images: ")
    gt_images_gray = [np.mean(img, axis=2) for img in tqdm(gt_images)]
    render_images_gray = [np.mean(img, axis=2) for img in tqdm(render_images)]

    # calculate the mean squared error        
    print("calculating mse and mae: ")
    mse_images = []
    mae_images = []
    if mse_enabled:
        mse_images = [mse(gt, render) for gt, render in tqdm(zip(gt_images_gray, render_images_gray))]
    if mae_enabled:
        mae_images = [mae(gt, render) for gt, render in tqdm(zip(gt_images_gray, render_images_gray))]

    #find a mae not equal to 0

    # for i in mae_images:
        # if i != 0:


    #create the save directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "mse"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "mae"), exist_ok=True)
    # save the images
    print("saving mse images: ")
    for i in tqdm(range(len(mse_images))):
        str_i = str(i)
        if len(str_i) != 6:
            str_i = "0" * (6-len(str_i)) + str_i
        plt.imsave(os.path.join(os.path.join(save_path, f"mse"), f'{str_i}.png'), mse_images[i], cmap='jet')

    print("saving mae images: ")
    for i in tqdm(range(len(mae_images))):
        str_i = str(i)
        if len(str_i) != 6:
            str_i = "0" * (6-len(str_i)) + str_i
        plt.imsave(os.path.join(os.path.join(save_path, f"mae"), f'{str_i}.png'), mae_images[i], cmap='jet')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Color Map Generation')
    parser.add_argument('--gt_path', type=str, help='Path to the ground truth images')
    parser.add_argument('--render_path', type=str, help='Path to the rendered images')
    parser.add_argument('--save_path', type=str, default="./" , help='Path to save the color maps')
    parser.add_argument('--mse', type=bool, default=True, help='Generate MSE images')
    parser.add_argument('--mae', type=bool, default=True, help='Generate MAE images')
    args = parser.parse_args()

    get_color_map(args.gt_path, args.render_path, args.save_path, args.mse, args.mae)

    