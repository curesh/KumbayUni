import cv2 as cv
import numpy as np
import os
import sys

class Anon():

    # Initialize instance variables
    def __init__(self, vid_dir):
        print("Constructing")
        vid = cv.VideoCapture(vid_dir)
        self.vid = vid
        frames = []
        while vid.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        self.frames = frames
        
    def __del__(self):
        print("Destructing")
        self.vid.release()
        
    # Preprocess the video (get frames, etc.)
    def _preprocess(self):
        pass
    
    # Static face anonymization
    def anon_static(self):
        pass

    def play_vid(self):
        for frame in self.frames:
            cv.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
def main():
    vid_dir = os.getcwd() + ""
    anon = Anon(vid_dir)
    anon.play_vid()

if __name__ == "__main__":
    main()
    
