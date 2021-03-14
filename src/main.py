import argparse
from video_processor import VideoProcessor


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, help="path to input image or video")
ap.add_argument("-c", "--config", required=True, help="configuration path")

args = vars(ap.parse_args())

if __name__ == '__main__':
    processor = VideoProcessor(args)
    try:
        processor.start()
    except KeyboardInterrupt:
        VideoProcessor.close()






