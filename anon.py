import cv2 as cv
import numpy as np
import os
import sys
# import dlib
# import face_recognition

ZOOM_WIDTH = 244
ZOOM_HEIGHT = 138

class Anon():

    # Initialize instance variables
    def __init__(self, vid_dir):
        print("Constructing")
        vid = cv.VideoCapture(vid_dir)
        self.vid = vid
        frames = []
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.scale=1
                if frame.shape[0] > 1000 or frame.shape[1] > 1000:
                    self.scale = 1000/max(frame.shape[0], frame.shape[1])
                    width = int(frame.shape[1] * self.scale)
                    height = int(frame.shape[0] * self.scale)
                    frame = cv.resize(frame, (width, height))
                frames.append(frame)
            else:
                break
        self.frames = frames
        scale = 1
        if len(self.frames) > 1000:
            scale = int(len(self.frames)/1000)
        self.frames_step = self.frames[::scale]
        self.shape = np.shape(self.frames)
        self.shape_step = np.shape(self.frames_step)
        self.scale_frame = scale
        
        print("Done constructing")
    def __del__(self):
        print("Destructing")
        self.vid.release()
        
    # Preprocess the video (get frames, etc.)
    def _preprocess(self):
        pass
    
    def _detect_shapes(self, cnt):
        pass
    

    def _find_zoom(self, test_frame):
        # test_frame = self.frames[int(self.shape[0]*0.78)]
        area_zoom = ZOOM_HEIGHT*ZOOM_WIDTH*(self.scale**2)
        gray = cv.cvtColor(test_frame, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            # if w*h*2 > area_zoom:
            #     print("wh2: ", w*h*2, ", ", "area_zoom: ", area_zoom)
            #     return [0, 0, self.shape[2], self.shape[1]]
            x_new = int(max(0, x-0.75*w))
            y_new = int(max(0, y-0.5*h))
            w_new = int(min(self.shape[2], 2.75*w))
            h_new = int(min(self.shape[1], 2.5*h))
            cv.rectangle(test_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # cv.imshow("zoom frame: ", test_frame)

            # cv.waitKey(0)
            # cv.destroyAllWindows()
            return [x_new, y_new, w_new, h_new, x,y,w,h]


        
    # Static face anonymization
    def anon_static(self):
        # print("self.frames: ", self.frames)
        for i, test_frame in enumerate(self.frames_step):
            if i%50 == 0:
                print("i: ", i)
            rect = self._find_zoom(test_frame)
            if rect is None:
                continue
            good_rect = rect[4:]
            rect = rect[0:4]

            for j in range(i, i+self.scale_frame):
                # self.frames[j] = cv.rectangle(self.frames[j], (0,0), (self.shape[2], self.shape[1]), (255,0,0), 2)
                # print("j: ", j)
                # print("self shape", self.shape)
                self.frames[j] = cv.rectangle(self.frames[j], (good_rect[0], good_rect[1]), (good_rect[0]+good_rect[2], good_rect[1]+good_rect[3]), (0,255,0), 4)
                self.frames[j] = cv.rectangle(self.frames[j], (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,0,0), -1)
                self.frames[j] = cv.rectangle(self.frames[j], (good_rect[0], good_rect[1]), (good_rect[0]+good_rect[2], good_rect[1]+good_rect[3]), (0,255,0), 4)
                # self.frames[j] = cv.rectangle(self.frames[j], (0, 0), (int(ZOOM_WIDTH*self.scale), int(ZOOM_HEIGHT*self.scale)), (255,255,0), 4)
                # cv.imshow("check self frames", self.frames[j])
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                
    def play_vid(self, frames):
        print("len frames: ", len(frames))
        cv.imshow("frame", frames[0])
        cv.waitKey(0)
        for frame in frames :
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
def main():
    vid_dir = os.getcwd() + "/test_data/test1.mp4"
    anon = Anon(vid_dir)
    anon.anon_static()
    anon.play_vid(anon.frames)
    print("Shape frames:", np.shape(anon.frames))
    print("One frame shape: ", np.shape(anon.frames[0]))


if __name__ == "__main__":
    main()
    
