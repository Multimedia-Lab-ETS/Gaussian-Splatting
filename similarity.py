#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in tqdm(os.listdir(renders_dir)):
        # print(fname)
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names
    
def evaluate(args):

    ssims = []
    psnrs = []
    lpipss = []

    train_ssims = []
    train_psnrs = []
    train_lpipss = []

    render_dir = args.render_path[0]
    gt_dir = args.gt_path[0]
    print(type(render_dir))

    renders, gts, image_names = readImages(render_dir, gt_dir)

    for idx in tqdm(range(len(renders)), desc="Testing Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))


    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    per_view_dict = {}

    full_dict = {}


    full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    test_dir = render_dir
    #  graph for SSIM
    plt.figure()
    plt.plot(torch.tensor(ssims).tolist())
    plt.title("SSIM")
    plt.xlabel("View")
    plt.ylabel("SSIM")
    plt.savefig(os.path.join(test_dir,"ssim.png"))
    
    #  graph for PSNR
    plt.figure()
    plt.plot(torch.tensor(psnrs).tolist())
    plt.title("PSNR")
    plt.xlabel("View")
    plt.ylabel("PSNR")
    plt.savefig(os.path.join(test_dir,"psnr.png"))

    #  graph for LPIPS
    plt.figure()
    plt.plot(torch.tensor(lpipss).tolist())
    plt.title("LPIPS")
    plt.xlabel("View")
    plt.ylabel("LPIPS")
    plt.savefig(os.path.join(test_dir,"lpips.png"))

    with open(test_dir / "metrics.json", "w") as f:
        json.dump(full_dict, f)

    with open(test_dir / "per_view_metrics.json", "w") as f:
        json.dump(per_view_dict, f)


    print("")
    print("Results saved in", render_dir)

        
        #do for train
            
if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gt_path', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--render_path', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    # print(args)
    evaluate(args)
