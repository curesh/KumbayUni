import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
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
        self.frames = frames[:400]
        scale = 1
        if len(self.frames) > 1000:
            scale = int(len(self.frames)/1000)
        self.frames_step = self.frames[::scale]
        self.shape = np.shape(self.frames)
        print("Shape: ", self.shape)
        self.shape_step = np.shape(self.frames_step)
        self.scale_frame = scale
        self.fps = vid.get(cv.CAP_PROP_FPS)

        
        print("Done constructing")
    def __del__(self):
        print("Destructing")
        self.vid.release()
        
    # Preprocess the video (get frames, etc.)
    def _preprocess(self):
        pass
    
    # Check if two rectangles are overlapping
    def _rect_overlap(self, rect, other_rect):
        # If one rectangle is on left side of other 
        if(rect[0] >= other_rect[2]+other_rect[0] or other_rect[0] >= rect[2]+rect[0]):
            return False
    
        # If one rectangle is above other 
        if(rect[1] >= other_rect[3]+other_rect[1] or other_rect[1] >= rect[3]+rect[1]): 
            return False
    
        return True
    
    # Given an array of rectangles, return a new array of rectangles, with only those that are non overlapping
    # Given an three arrays of rectangles, return only
    def _rect_compare(self, all_rects):
        # all_rects_len = [len(rects1), len(rects2), len(rects3)]
        # all_rects = [rects1, rects2, rects3]
        # ind_sort = np.argsort(all_rects_len)[::-1]
        # all_rects = [all_rects[index] for index in ind_sort]
        
        non_overlap = []
        # overlapped = True
        # # for i in range(max(len(rects1), len(rects2), len(rects3))):
        # #     if i < len(all_rects[1]):

        # while overlapped:
        #     for rect in non_overlap:
        #         if self._rect_overlap(all_rects[i], all_rects[j]):
        #             if all_rects[i][2]*all_rects[i][3] >= all_rects[j][2]*all_rects[j][3]:
        #                  non_overlap.append(all_rects[i])
        #             else:
        #                  non_overlap.append(all_rects[j])
                    #  non_overlap.append()
        return  non_overlap
    
    # This function is used for quantizing a rectangle for the purposes of generating the graph over time
    def _rect_func(self, rect):
        alpha = 0.1
        beta = 1-alpha
        
        rect  = [rect[0]/self.shape[2], rect[1]/self.shape[1], rect[2]/self.shape[2], rect[3]/self.shape[1]]
        
        # rect_val = alpha*(rect[0] + rect[1])/2
        # rect_val += beta*(rect[2]*rect[3])
        
        rect_val = (rect[2]+rect[3])/2
        return rect_val
    
    # Orders rectangle array by rects from largest to smallest
    def _find_greatest_rect(self, rects):
        if not rects:
            return None
        ans = rects[0]
        for rect in rects:
            if self._rect_func(rect) > self._rect_func(ans):
                ans = rect
        return ans
        
    def _order_rects(self, rects, quants=None):
        if rects is None:
            return [[0,0,0,0]], [0]
        if quants is None:
            quants = []
            for rect in rects:
                quants.append(self._rect_func(rect))
        ind_sort = np.argsort(quants)[::-1]
        rects = [rects[index] for index in ind_sort]
        quants.sort(reverse=True)
        return rects, quants
    
    # Additional feature you want to add, is that if a large frontal box isn't there for a given, frame 
    # (yet, there was a frontal box within the last 8 seconds, it should check either profile)
    
    # Test cases not handled yet
    # 1. At a single frame, there is no face detected
    # 2. At a single frame there are multiple true faces detected
    # 3. At a single frame, there are multiple true and some false faces detected
    
    # Ideas that need implementing
    # 1. Check if its a real world image or a screenrecording by doing some stastics on curr_frame, or maybe some machine learning model
    
    def _find_zoom(self):
        #  curr_frame = self.frames[int(self.shape[0]*0.78)]
        area_zoom = ZOOM_HEIGHT*ZOOM_WIDTH*(self.scale**2)
        
        quant_rect_tot = []
        rects_tot = []
        i_last_large_head = 0
        max_frames_large_head = int(8*self.fps/self.scale_frame)
        prof_face_cascade = cv.CascadeClassifier('/System/Volumes/Data/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/cv2/data/haarcascade_profileface.xml')
        face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

        for i,  curr_frame in enumerate(self.frames_step):
            if i%50 == 0:
                print("i: ", i)     
            
            gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
            # prof_face_cascade = cv.CascadeClassifier('./lbpcascade_profileface.xml')
            # detectMultiScale(...) Params:
            # Image---
            # scaleFactor--- To detect large and small faces alike, the program repeatedly
            # downsamples the image and checks for faces of a specific size. This parameter
            # determines the rate at which the downsampling is done. A larger number means
            # faster runtime, but a higher probability of detection misses.
            # minNeighbors--- Higher value results in less detections but with higher quality.
            # 3~6 is a good value for it.
            front_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangle around the faces
            for (x,y,w,h) in front_faces:
                cv.rectangle(curr_frame, (x,y), (x+w, y+h), (255,0,0), 2)

            rects, quant_rect = self._order_rects(list(front_faces))
            if not rects:
                rects = [[0,0,0,0]]
                quant_rect = [0]
            
            if rects[0][3] > ZOOM_HEIGHT:
                i_last_large_head = 0
            if (rects[0][3] < ZOOM_HEIGHT) and (i_last_large_head < max_frames_large_head):
                print("curr i: ", i)
                print("rects: ", rects)
                right_side_faces = prof_face_cascade.detectMultiScale(gray, 1.1, 4)
                left_side_faces = prof_face_cascade.detectMultiScale(cv.flip(gray, 1), 1.1, 4)
                left_side_faces = [[self.shape[2]-curr_rect[0], curr_rect[1], curr_rect[2], curr_rect[3]] for curr_rect in left_side_faces]
                both_side_faces = list(right_side_faces).extend(list(left_side_faces))
                max_both_rect = self._find_greatest_rect(both_side_faces)
                print("Both side faces: ", both_side_faces)
                print("Max both rect: ", max_both_rect)
                cv.imshow("frame", curr_frame)
                cv.waitKey(0)
                if max_both_rect and (max_both_rect[3] > ZOOM_HEIGHT):
                    rects.insert(0, max_both_rect)
                    quant_rect.insert(0, self._rect_func(max_both_rect))
                    cv.rectangle(curr_frame, (max_both_rect[0], max_both_rect[1]),
                                (max_both_rect[0]+max_both_rect[2], max_both_rect[1]+max_both_rect[3]),
                                (0,255,0), 2)
                    

            quant_rect_tot.append(quant_rect)
            rects_tot.append(rects)
            i_last_large_head += 1
            
        return quant_rect_tot, rects_tot
        
    # Static face anonymization
    def anon_static(self):
        # print("self.frames: ", self.frames)
        quant_rect_tot, rects_tot = self._find_zoom()
        first_elem_quant_rect = [vec[0] for vec in quant_rect_tot]
        plt.plot(first_elem_quant_rect)
        plt.show()
                
    def play_vid(self, frames):
        print("len frames: ", len(frames))
        cv.imshow("frame", frames[0])
        cv.waitKey(0)
        for frame in frames :
            cv.imshow('frame', frame)
            if cv.waitKey(4) & 0xFF == ord('q'):
                break

def main():
    vid_dir = os.getcwd() + "/test_data/test2.mp4"
    anon = Anon(vid_dir)
    anon.anon_static()
    for i in range(20):
        anon.play_vid(anon.frames)
    
    print("Shape frames:", np.shape(anon.frames))
    print("One frame shape: ", np.shape(anon.frames[0]))

def main_test():
    vid_dir = os.getcwd() + "/test_data/test2.mp4"
    anon = Anon(vid_dir)
    print("rect overlap: ", anon._rect_compare([[0, 20, 10, 10], [9, 30, 20, 20], [30, 30, 1, 1]]))


if __name__ == "__main__":
    main()
    
