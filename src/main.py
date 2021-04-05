import argparse
from video_processor import VideoProcessor


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, help="path to input image or video")
ap.add_argument("-c", "--config", required=True, help="configuration path")
ap.add_argument("--use-server", type=str2bool,  nargs='?', const=True, required=False, help="use http server", default=False)
ap.add_argument("-ip", "--ip", required=False, help="http server ip")
ap.add_argument("-p", "--port", required=False, help="http server port")

args = ap.parse_args()

if __name__ == '__main__':
    processor = VideoProcessor(args)
    try:
        processor.start()
    except KeyboardInterrupt:
        VideoProcessor.close()






