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
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names
    
def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}
            
            test_dir = Path(scene_dir) / "test"
            train_dir = Path(scene_dir) / "train"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # train
                method_dir = train_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders_train, gts_train, image_names_train = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                train_ssims = []
                train_psnrs = []
                train_lpipss = []

                for idx in tqdm(range(len(renders)), desc="Testing Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                
                for idx in tqdm(range(len(renders_train)), desc="Training Metric evaluation progress"):
                    train_ssims.append(ssim(renders_train[idx], gts_train[idx]))
                    train_psnrs.append(psnr(renders_train[idx], gts_train[idx]))
                    train_lpipss.append(lpips(renders_train[idx], gts_train[idx], net_type='vgg'))


                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                print("  Train SSIM : {:>12.7f}".format(torch.tensor(train_ssims).mean(), ".5"))
                print("  Train PSNR : {:>12.7f}".format(torch.tensor(train_psnrs).mean(), ".5"))
                print("  Train LPIPS: {:>12.7f}".format(torch.tensor(train_lpipss).mean(), ".5"))
                print("")


                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})


                # for train
                full_dict[scene_dir+"_train"][method].update({"SSIM": torch.tensor(train_ssims).mean().item(),
                                                        "PSNR": torch.tensor(train_psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(train_lpipss).mean().item()})
                per_view_dict[scene_dir+"_train"][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(train_ssims).tolist(), image_names_train)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(train_psnrs).tolist(), image_names_train)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(train_lpipss).tolist(), image_names_train)}})
                
                #  graph for SSIM
                plt.figure()
                plt.plot(torch.tensor(ssims).tolist())
                plt.title("SSIM")
                plt.xlabel("View")
                plt.ylabel("SSIM")
                plt.savefig(test_dir / "ssim.png")
                
                #  graph for PSNR
                plt.figure()
                plt.plot(torch.tensor(psnrs).tolist())
                plt.title("PSNR")
                plt.xlabel("View")
                plt.ylabel("PSNR")
                plt.savefig(test_dir / "psnr.png")

                #  graph for LPIPS
                plt.figure()
                plt.plot(torch.tensor(lpipss).tolist())
                plt.title("LPIPS")
                plt.xlabel("View")
                plt.ylabel("LPIPS")
                plt.savefig(test_dir / "lpips.png")

                #  graph for SSIM train
                plt.figure()
                plt.plot(torch.tensor(train_ssims).tolist())
                plt.title("SSIM")
                plt.xlabel("View")
                plt.ylabel("SSIM")
                plt.savefig(train_dir / "ssim.png")

                #  graph for PSNR train
                plt.figure()
                plt.plot(torch.tensor(train_psnrs).tolist())
                plt.title("PSNR")
                plt.xlabel("View")
                plt.ylabel("PSNR")
                plt.savefig(train_dir / "psnr.png")

                #  graph for LPIPS train
                plt.figure()
                plt.plot(torch.tensor(train_lpipss).tolist())
                plt.title("LPIPS")
                plt.xlabel("View")
                plt.ylabel("LPIPS")
                plt.savefig(train_dir / "lpips.png")


        

            with open(scene_dir + "/results_test.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_test.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

    
            with open(scene_dir + "/results_train.json", 'w') as fp:
                json.dump(full_dict[scene_dir+"_train"], fp, indent=True)
            with open(scene_dir + "/per_view_train.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir+"_train"], fp, indent=True)

            print("")
            print("Results saved in", scene_dir)

        
        #do for train
            

        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
