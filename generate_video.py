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

import torch
from scene import Scene
from scene.cameras import Camera
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

g_index = 0
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, frame):
    global g_index
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_vids")
    print(render_path)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_vids")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        # gt = view.original_image[0:3, :, :]
        index = g_index
        # index = frame*24 + idx
        index = str(index)
        if len(index) != 6:
            index = "0" * (6-len(index)) + index
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{index}") + ".png")
        g_index+= 1
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(index) + ".png"))
    

def lerp_cameras(cam1, cam2, iterations):
    cameras = []
    for i in tqdm(range(iterations)):
        alpha = i/iterations
        new_R = cam1.R * (1 - alpha) + cam2.R * (alpha)
        new_T = cam1.T * (1 - alpha) + cam2.T * (alpha)
        cam = Camera(0, new_R, new_T, cam1.FoVx, cam1.FoVy, cam1.original_image, None, cam1.image_name, 0)
        # print(alpha)
        
        cameras.append(cam)
    return cameras
    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, fps : int = 24):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # get all the cameras
        print("lerping camera")
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        last_index = 0
        for i in tqdm(range(10,len(all_cameras),10)):
            cam_set = lerp_cameras(all_cameras[last_index], all_cameras[i], fps)
            last_index = i
    
            render_set(dataset.model_path, "test", scene.loaded_iter, cam_set, gaussians, pipeline, background, i)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--fps",  default=24, type=int)
    # parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.fps)