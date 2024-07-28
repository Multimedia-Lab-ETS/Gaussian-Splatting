from ffmpy import FFmpeg
import os
import argparse

# python ./film.py -video_path ./MVI_4605.MOV -output_path ./grass5 -fps 10

# take 5 frames per second from a video
def extract_frames(video_path, output_path, fps=5):
    # convert the command  ffmpeg -i .\a.mkv -qscale:v 1 -qmin 1 -vf fps=0.2 %0.4d.jpg to python
    ff = FFmpeg(
        inputs={video_path: None},
        outputs={os.path.join(output_path, '%04d.jpg'): f'-qscale:v 1 -qmin 1 -vf fps={fps}'}
    )
    ff.run()

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='path to the video')
    parser.add_argument('--output_path', type=str, help='path to the output folder')
    parser.add_argument('--fps', type=float, default=5, help='frames per second')
    args = parser.parse_args()

    # extract frames
    extract_frames(args.video_path, args.output_path, args.fps)

if __name__ == '__main__':
    main()
