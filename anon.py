import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

ZOOM_WIDTH = 244
ZOOM_HEIGHT = 138

class Anon():

    # Initialize instance variables
    def __init__(self, vid_dir=None):
        if vid_dir == None:
            print("Test_main Activated")
            self.vid = None
            self.scale = 1
            self.frames = []
            self.frames_step = []
            self.shape = [1,1,1]
            self.shape_step = [1,1,1]
            self.scale_frame = 1
            self.fps = 1
            return None
        
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
        print("Shape: ", self.shape)
        self.shape_step = np.shape(self.frames_step)
        self.scale_frame = scale
        self.fps = vid.get(cv.CAP_PROP_FPS)
    
        self.standard_rect = [int(self.shape[2]/3), int(self.shape[1]/3),
                              int(self.shape[2]/3), int(self.shape[1]/3)]
        print("Standard rect: ", self.standard_rect)
        print("Done constructing")
    def __del__(self):
        print("Destructing")
        if self.vid is not None:
            self.vid.release()
        
    # Preprocess the video (get frames, etc.)
    def _preprocess(self):
        pass
    
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
        
        # The 4 here is the number of seconds that it will check profile faces for (since detecting the last large face)
        max_frames_large_head = int(4*self.fps/self.scale_frame)
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
            rects, quant_rect = self._order_rects(list(front_faces))
            if not rects:
                rects = [[0,0,0,0]]
                quant_rect = [0]
            
            if rects[0][3] > ZOOM_HEIGHT:
                i_last_large_head = 0
            
            if (rects[0][3] < ZOOM_HEIGHT) and (i_last_large_head < max_frames_large_head):
                right_side_faces = prof_face_cascade.detectMultiScale(gray, 1.1, 4)
                left_side_faces = prof_face_cascade.detectMultiScale(cv.flip(gray, 1), 1.1, 4)
                left_side_faces = [[self.shape[2]-curr_rect[0]-curr_rect[2], curr_rect[1], curr_rect[2], curr_rect[3]] for curr_rect in left_side_faces]
                if not list(right_side_faces):
                    both_side_faces = left_side_faces
                else:
                    both_side_faces = left_side_faces.extend([list(elem) for elem in right_side_faces])
                max_both_rect = self._find_greatest_rect(both_side_faces)
                if max_both_rect and (max_both_rect[3] > ZOOM_HEIGHT):
                    rects.insert(0, max_both_rect)
                    quant_rect.insert(0, self._rect_func(max_both_rect))

            quant_rect_tot.append(quant_rect)
            rects_tot.append(rects)
            i_last_large_head += 1
            
        return quant_rect_tot, rects_tot
    
    def _draw_rects(self, rects):
        for i, rects_frame in enumerate(rects):
            for rect in rects_frame:
                if rect[2] == 0:
                    rects[i].remove(rect)
                    continue
                p1 = (int(max(0,rect[0]-0.25*rect[2])), int(max(0,rect[1]-0.25*rect[3])))
                p2 = (int(min(self.shape[2],rect[0]+1.25*rect[2])), int(min(self.shape[1],rect[1]+1.25*rect[3])))
                cv.rectangle(self.frames[i], p1, p2, (255, 255, 0), 4)
                # print("rect for subface: ", rect)
                # print("p1 and p2: ", p1, ", ", p2)
                sub_face = self.frames[i][p1[1]:min(p2[1],self.shape[1]),p1[0]:min(p2[0], self.shape[2])]
                # apply a gaussian blur on this new recangle image
                sub_face = cv.GaussianBlur(sub_face, (171, 171), 60)
                # merge this blurry rectangle to our final image
                self.frames[i][p1[1]:min(p2[1],self.shape[1]),p1[0]:min(p2[0], self.shape[2])] = sub_face
        
    # Smooth out the plot of the largest rectangle areas across all the frames. 
    # Do this adding a large rectangle to a given frame, if there was x (x may just be 1) large
    # rectangles at most y seconds ago (y will probably be about 4)
    def smooth_largest(self, rects, quants, x, y):
        threshold_frames = int(self.fps*y/self.scale_frame)
        
        num_large = 1
        frames_since_large = 0
        avg_large_rect = np.array([0,0,0,0])
        for i in range(len(rects)):
            # NOTE:This 0.8 here is an important paramenter and should be supplied from elsewhere
            if rects[i][0][3] > max(int(avg_large_rect[3]*0.8),ZOOM_HEIGHT):
                if avg_large_rect[2] == 0:
                    avg_large_rect = rects[i][0].copy()
                avg_large_rect = np.add(avg_large_rect, rects[i][0])/2
                num_large += 1
                frames_since_large = 0
            elif frames_since_large < threshold_frames and num_large >= x:
                # print("Current avg large rect: ", avg_large_rect)
                if avg_large_rect[2] != 0:
                    insert_rect = [avg_large_rect[0]-frames_since_large, avg_large_rect[1]-frames_since_large, avg_large_rect[2]+frames_since_large, avg_large_rect[3]+frames_since_large]
                    rects[i].insert(0, insert_rect)
                    quants[i].insert(0, self._rect_func(insert_rect))
                else:
                    rects[i].insert(0, self.standard_rect)
                    quants[i].insert(0, self._rect_func(self.standard_rect))
                    rects[i], quants[i] = self._order_rects(rects[i])
                # cv.rectangle(self.frames[i], p1, p2, (255,255,0), 2)
            else:
                avg_large_rect = [0,0,0,0]
                num_large = 0
            frames_since_large += 1
        return quants, rects
    
    # Static face anonymization
    def anon_static(self):
        # print("self.frames: ", self.frames)
        quant_rect_tot, rects_tot = self._find_zoom()
        # print("rects_tot: ", rects_tot)
        # print("quants rect tot: ", quant_rect_tot)
        first_elem_quant_rect = [vec[0] for vec in quant_rect_tot]
        plt.figure()
        plt.plot(first_elem_quant_rect, color='b')
        quant_rect_tot, rects_tot = self.smooth_largest(rects_tot, quant_rect_tot, 1, 4)
        quant_rect_tot, rects_tot = self._remove_overlapping(rects_tot)
        # print("rects after remove overlapping: ", rects_tot)
        self._draw_rects(rects_tot)
        first_elem_quant_rect = [vec[0] for vec in quant_rect_tot]
        plt.figure()
        plt.plot(first_elem_quant_rect, color='r')
        plt.show()

    # Return the area of overlapped region for two rectangles
    # If they don't overlap, return -1
    def _overlap_area(self, rect, other_rect):
        width = min(rect[0]+rect[2],other_rect[0]+other_rect[2])-max(rect[0],other_rect[0])
        if width <= 0:
            return -1
        height = min(rect[1]+rect[3],other_rect[1]+other_rect[3])-max(rect[1],other_rect[1])
        if height <= 0:
            return -1    
        return width*height
        
    # For each frame, return a new subset of maximized (by area) rectangles where non overlap
    def _remove_overlapping(self, rects):
        quants = []
        for i, rects_frame in enumerate(rects):
            
            # This is the array of rectangles that you need to empty. It is sorted from greatest to least
            check_rects = rects_frame.copy()
            nonoverlap_rects = []
            nonoverlap_quants = []
            while check_rects:
                largest_rect = check_rects[0]
                area_largest = largest_rect[2]*largest_rect[3]
                good = True
                for rect in nonoverlap_rects:
                    # area_rect = rect[2]*rect[3]
                    overlap = self._overlap_area(rect, largest_rect)
                    if area_largest < 10*overlap or area_largest == 0:
                        # If the overlap is not really small, then consider it an overlap
                        good = False
                        break
                if good:
                    nonoverlap_rects.append(largest_rect)
                    nonoverlap_quants.append(self._rect_func(largest_rect))
                check_rects.remove(largest_rect)
            rects[i] = nonoverlap_rects
            quants.append(nonoverlap_quants)
            
        return quants, rects
    

                
                
    def play_vid(self, frames):
        print("len frames: ", len(frames))
        cv.imshow("frame", frames[0])
        cv.waitKey(0)
        for frame in frames :
            cv.imshow('frame', frame)
            if cv.waitKey(4) & 0xFF == ord('q'):
                break

def main():
    vid_dir = os.getcwd() + "/test_data/test1.mp4"
    anon = Anon(vid_dir)
    anon.anon_static()
    for i in range(20):
        anon.play_vid(anon.frames)
    
    print("Shape frames:", np.shape(anon.frames))
    print("One frame shape: ", np.shape(anon.frames[0]))

def main_test():
    vid_dir = os.getcwd() + "/test_data/test1.mp4"
    anon = Anon()
    print("greatest rect: ", anon._remove_overlapping([[[0, 20, 10, 10], [9, 30, 20, 20], [20, 30, 1, 1]]]))
    print("overlap rect: ", anon._overlap_area([9, 30, 20, 20], [30, 30, 1, 1]))


if __name__ == "__main__":
    main()
    
