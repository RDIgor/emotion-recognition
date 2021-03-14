import cv2


class VideoWriter:
    def __init__(self, output_file, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 15, (frame_width, frame_height))

    def write(self, frame):
        resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        self.out.write(resized)

    def release(self):
        self.out.release()
