import os
import argparse


def video_to_photo(repo_dir: str, video_path : str, output_path: str = "./input",  fps_reading : int = 5, env_name: str = "gaussian_splatting"):

    script_loc = os.path.join(repo_dir, "video_to_image.py")
    #Calling the Script
    os.system(f" python {script_loc} --video_path {video_path} --output_path {output_path} --fps {fps_reading}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_dir', type=str, help='Location of the original repo', default= "./")
    parser.add_argument('--video_path', type=str, help='path to the input video', default= "./vid.mp4")
    parser.add_argument('--output_path', type=str, help='path to the output image folder', default= "./input")
    parser.add_argument('--fps_reading', type=float, default=5, help='frames per second for reading the video')
    args = parser.parse_args()

    print("Processing video to images")
    video_to_photo(args.repo_dir, args.video_path, args.output_path, args.fps_reading)
    
