from ffmpy import FFmpeg
import os
import argparse

# python ./film.py -video_path ./MVI_4605.MOV -output_path ./grass5 -fps 10

# take 5 frames per second from a video
def extract_frames(input_path, output_path, fps=24):
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory {input_path} does not exist")

    if not os.access(input_path, os.R_OK):
        raise PermissionError(f"Read permission denied for input directory {input_path}")
    

    ff = FFmpeg(
    inputs={os.path.join(input_path, '%06d.png'): f'-framerate {fps}'},
    outputs={output_path: f'-c:v libx265 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r {fps} -pix_fmt yuv420p'}
)
    ff.run()

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to the input video')
    parser.add_argument('--output_path', type=str, help='path to the output mp4', default='out.mp4')
    parser.add_argument('--fps', type=float, default=5, help='frames per second')
    args = parser.parse_args()

    # extract frames
    extract_frames(args.input_path, args.output_path, args.fps)

if __name__ == '__main__':
    main()
