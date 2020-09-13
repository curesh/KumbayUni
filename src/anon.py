# anon.py by Chandra Suresh

# Ideas that need implementing
# 1. Check if its a real world image or a screenrecording by doing some stastics on curr_frame, or maybe some machine learning model

import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import time

# Global Constants

# The dimensions for a zoom box are constant regardless of screen resolution in screen recordings
ZOOM_WIDTH = 244
ZOOM_HEIGHT = 138
FNULL = open(os.devnull, 'w')
t0 = time.time()

# Class used for storing the video frames, and relavent metadata
class Anon():

    # Initialize instance variables
    def __init__(self, vid_dir=None):
        path_dir = Path(vid_dir)
        # Alternate "Constructor" used for testing
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
            self.vid_dir = vid_dir
            return None
        
        print("Constructing")
        # Initialize video capture
        path_dir = Path(vid_dir)
        # Downsample video file using ffmpeg
        command_get_frames = "ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 " + vid_dir
        command_get_shape = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 " + vid_dir
        command_rename = ["mv", os.path.join(path_dir.parent, "lowres"+path_dir.name), vid_dir]

        num_frames = subprocess.run(command_get_frames.split(), stdout=subprocess.PIPE).stdout.decode("utf-8")
        print("Num frames: ", num_frames)
        width, height = subprocess.run(command_get_shape.split(), stdout=subprocess.PIPE).stdout.decode("utf-8").split("x")
        print("Before Width height: ", width, ", ", height)
        width, height = (int(width), int(height))
        self.scale  = 1
        if width > 1000 or height > 1000:
            print("Supposedly adjusting said properties")
            self.scale = 1000/max(width, height)
            width = int(width * self.scale)
            height = int(height * self.scale)
            
            command_lowres = "ffmpeg -y -i " + vid_dir + " -s " + str(width) + "x" + str(height) + " -vcodec mpeg4 -q:v 20 -acodec copy " + os.path.join(path_dir.parent, "lowres"+path_dir.name)
            subprocess.run(command_lowres.split())
            os.remove(vid_dir)
            subprocess.run(command_rename, stdout=FNULL, stderr=subprocess.STDOUT)
        print("After downsampling Width height: ", width, ", ", height)
        
        self.vid = cv.VideoCapture(vid_dir)
        t_videoCapture = time.time()
        print("TIME after loading in videoCapture: ", t_videoCapture-t0)
        
        self.fps = self.vid.get(cv.CAP_PROP_FPS)
        print("Video FPS: ", self.fps)
        frames = []
        # Collect the frames
        while self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                    # frame = cv.resize(frame, (width, height))
                frames.append(frame)
            else:
                break
        self.frames = frames
        scale = 1

        # Perform analysis on a lower number of frames, then interpolate in post-processing
        if len(self.frames) > 100:
            scale = int(len(self.frames)/100)

        # Store various metadata
        self.vid_dir = vid_dir
        self.frames_step = self.frames[::scale]
        self.shape = np.shape(self.frames)
        print("Shape: ", self.shape)
        self.shape_step = np.shape(self.frames_step)
        self.scale_frame = scale
        self.standard_rect = [int(self.shape[2]/3), int(self.shape[1]/3),
                              int(self.shape[2]/3), int(self.shape[1]/3)]

        print("Done constructing")

    # Destructor
    def __del__(self):
        print("Destructing")
        if self.vid is not None:
            self.vid.release()
        
    # Preprocess the video (get frames, etc.)
    def _preprocess(self):
        pass
    
    # This function is used for quantizing a rectangle for the purposes of generating the graph over time of the size of the rectangles
    def _rect_func(self, rect):
        rect  = [rect[0]/self.shape[2], rect[1]/self.shape[1], rect[2]/self.shape[2], rect[3]/self.shape[1]]
        rect_val = (rect[2]+rect[3])/2
        return rect_val
    
    # Find greatest recangle in an array of rectangle (greatest is defined by _rect_func(...))
    def _find_greatest_rect(self, rects):
        if not rects:
            return None

        ans = rects[0]
        
        for rect in rects:
            if self._rect_func(rect) > self._rect_func(ans):
                ans = rect
        return ans

    # Orders array of rectangles of greatest to least (as determined by _rect_func), and returns their quantized values
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

    # Function used to find the rectangles bordering faces in an image, using facial recognition
    # Returns an array of an array of rectangles per frame. Also returns another array (of the same shape)
    # with the corresponding quantizations
    def _find_zoom(self):
        area_zoom = ZOOM_HEIGHT*ZOOM_WIDTH*(self.scale**2)
        quant_rect_tot = []
        rects_tot = []
        i_last_large_head = 0
        
        # The 4 here is the number of seconds that it will check profile faces for (since detecting the last large face)
        max_frames_large_head = int(4*self.fps/self.scale_frame)

        # Load the face classifer
        #prof_face_cascade = cv.CascadeClassifier('/System/Volumes/Data/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/cv2/data/haarcascade_profileface.xml')
        #face_cascade = cv.CascadeClassifier('./assets/haarfiles/haarcascade_frontalface_default.xml')
        prof_face_cascade = cv.CascadeClassifier('static/assets/haarfiles/haarcascade_profileface.xml')
        face_cascade = cv.CascadeClassifier('static/assets/haarfiles/haarcascade_frontalface_default.xml')        
        # For every frame in the sampled video, detect frontal faces (and sometimes profile faces depending on certain parameters)
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
            
            # TODO: Do you need this order (redundant?)?
            rects, quant_rect = self._order_rects(list(front_faces))
            
            if not rects:
                rects = [[0,0,0,0]]
                quant_rect = [0]
            
            if rects[0][3] > ZOOM_HEIGHT:
                i_last_large_head = 0

            # If there has been a large face detected "recently" (as defined earlier) and there
            # is no large face now, run face detection for profile faces as well
            if (rects[0][3] < ZOOM_HEIGHT) and (i_last_large_head < max_frames_large_head):
                right_side_faces = prof_face_cascade.detectMultiScale(gray, 1.1, 4)
                left_side_faces = prof_face_cascade.detectMultiScale(cv.flip(gray, 1), 1.1, 4)

                # Since the face detection is built for right side_faces, we need to flip back the flipped rectangles that were created
                left_side_faces = [[self.shape[2]-curr_rect[0]-curr_rect[2], curr_rect[1], curr_rect[2], curr_rect[3]] for curr_rect in left_side_faces]
                
                if not list(right_side_faces):
                    both_side_faces = left_side_faces
                else:
                    both_side_faces = left_side_faces.extend([list(elem) for elem in right_side_faces])
                max_both_rect = self._find_greatest_rect(both_side_faces)
                if max_both_rect and (max_both_rect[3] > ZOOM_HEIGHT):
                    rects.insert(0, max_both_rect)
                    quant_rect.insert(0, self._rect_func(max_both_rect))

            # Add current frame's rectangles and quants to the main list
            quant_rect_tot.append(quant_rect)
            rects_tot.append(rects)
            i_last_large_head += 1
            
        return quant_rect_tot, rects_tot

    
    # Maybe check if they have similar areas, then check if they overlap a lot?
    def _check_avg(self, avg_small_rects, check_rect):
        for i, avg_rect in enumerate(avg_small_rects):
            norm_avg_area = avg_rect[2]*avg_rect[3]/(self.shape[1]*self.shape[2])
            norm_check_area = check_rect[2]*check_rect[3]/(self.shape[1]*self.shape[2])
            if np.abs(norm_avg_area-norm_check_area) < 0.3:
                if self._overlap_area(avg_rect[:-1], check_rect) > min(norm_check_area, norm_avg_area)/2:
                    return i, avg_rect
        return None, None
        
    # Smooth out the plot of the largest rectangle areas across all the frames. 
    # Do this adding a large rectangle to a given frame, if there were x (x may just be 1) large
    # rectangles at most y seconds ago (y will probably be about 4)
    def smooth_largest(self, rects, quants, x, y):
        threshold_frames = int(self.fps*y/self.scale_frame)
        
        num_large = 1
        frames_since_large = 0
        avg_large_rect = [0,0,0,0]
        avg_small_rects = []
        for i in range(len(rects)):
            avg_small_rects_curr = []
            avg_small_remove = []
            rects_curr = []
            # If this rectangle is large, then note that, and add it to the running average of large
            # rectangles
            # NOTE: The 0.8 here is an important parameter and should be supplied from elsewhere
            is_large = 0
            if rects[i][0][3] > max(int(avg_large_rect[3]*0.8),ZOOM_HEIGHT):
                is_large = 1
                if avg_large_rect[2] == 0:
                    avg_large_rect = rects[i][0].copy()
                avg_large_rect = list(np.add(avg_large_rect, rects[i][0])/2)
                num_large += 1
                frames_since_large = 0

            # If this isn't a large rectangle, but there has been one recently, apply the smoothing
            elif frames_since_large < threshold_frames and num_large >= x:
                if avg_large_rect[2] != 0:
                    insert_rect = [avg_large_rect[0]-frames_since_large, avg_large_rect[1]-frames_since_large, avg_large_rect[2]+frames_since_large, avg_large_rect[3]+frames_since_large]
                    rects_curr.append(insert_rect)
                else:
                    rects_curr.append(self.standard_rect)

            # Otherwise, keep the marked rectangle as is
            else:
                avg_large_rect = [0,0,0,0]
                num_large = 1
                
            frames_since_large += 1
            
            # For all rectangles in rects[i], add or update the corresponding rectangle in avg_small_rects
            for j in range(is_large, len(rects[i])):
                ind_small, small_rect_match = self._check_avg(avg_small_rects, rects[i][j])
                if ind_small is not None:
                    new_avg = np.add(avg_small_rects[ind_small][:-1], rects[i][j])/2
                    avg_small_rects[ind_small] = list(np.append(new_avg,0))
                else:
                    avg_small_rects_curr.append(list(np.append(rects[i][j].copy(),0)))
                    
            # For all rectangles in avg_small_rects but not in rects (and within threshold frames ago), apply smoothing algorithm
            for j, avg_rect in enumerate(avg_small_rects):
                if avg_rect[4] != 0 and avg_rect[4] < threshold_frames:
                    if avg_rect[2] != 0:
                        insert_rect = [avg_rect[0]-avg_rect[4], avg_rect[1]-avg_rect[4], avg_rect[2]+avg_rect[4], avg_rect[3]+avg_rect[4]]
                        rects_curr.append(insert_rect)
                elif avg_rect[4] != 0:
                    avg_small_remove.append(avg_rect)
                avg_small_rects[j][4] += 1
            # Add or remove the stored rects for this iteration "i"
            for elem in avg_small_rects_curr:
                avg_small_rects.append(elem)
            for elem in avg_small_remove:
                avg_small_rects.remove(elem)
            for elem in rects_curr:
                rects[i].append(elem)    
            rects[i], quants[i] = self._order_rects(rects[i])
            
        return quants, rects
    
    # Overarching anonymization function
    def anon_static(self):
        quant_rect_tot, rects_tot = self._find_zoom()
        t2 = time.time()
        print("TIME after _find_zoom(): ", t2-t0)
        first_elem_quant_rect = [vec[0] for vec in quant_rect_tot]

        # This plots the quantizations of the rectangles, before smoothign
        plt.figure()
        plt.plot(first_elem_quant_rect, color='b')

        quant_rect_tot, rects_tot = self.smooth_largest(rects_tot, quant_rect_tot, 1, 4)
        t3 = time.time()
        print("TIME after smooth forward: ", t3-t0)
        quant_rect_tot, rects_tot = self.smooth_largest(rects_tot[::-1], quant_rect_tot[::-1], 1, 4)
        quant_rect_tot = quant_rect_tot[::-1]
        rects_tot = rects_tot[::-1]
        t4 = time.time()
        print("TIME after smooth backwards: ", t4-t0)
        quant_rect_tot, rects_tot = self._remove_overlapping(rects_tot)
        t_remove_overlap = time.time()
        print("TIME after remove overlap: ", t_remove_overlap-t0)
        self._draw_rects(rects_tot)
        first_elem_quant_rect = [vec[0] for vec in quant_rect_tot]
        t5 = time.time()
        print("TIME after draw: ", t5-t0)
        # This plots the quantization of the rectangles after smoothing
        # plt.figure()
        # plt.plot(first_elem_quant_rect, color='r')
        # plt.show()

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
        
    # For each frame, return a new subset of maximized (by area) rectangles where none  overlap
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

                # True if not overlapping
                good = True
                for rect in nonoverlap_rects:
                    overlap = self._overlap_area(rect, largest_rect)
                    # If the overlap is not really small, then consider it an overlap
                    if area_largest < 10*overlap or area_largest == 0:
                        good = False
                        break
                if good:
                    nonoverlap_rects.append(largest_rect)
                    nonoverlap_quants.append(self._rect_func(largest_rect))
                check_rects.remove(largest_rect)
            rects[i] = nonoverlap_rects
            quants.append(nonoverlap_quants)            
        return quants, rects
    
    # Draw all the rectangles in the list to the corresping frames in the original array of frames
    def _draw_rects(self, rects):
        print("Beg of _draw_rects:")
        print("rects shape: ", len(rects))
        print("self.scale_frame: ", self.scale_frame)
        print("Big shape: ", self.shape)
        print("Small shape: ", self.shape_step)
        for i, rects_frame in enumerate(rects):
            j_start = i*self.scale_frame
            j_end = (i+1)*self.scale_frame
            for rect in rects_frame:
                if rect[2] == 0:
                    rects[i].remove(rect)
                    continue
                p1 = (int(max(0,rect[0]-0.25*rect[2])), int(max(0,rect[1]-0.25*rect[3])))
                p2 = (int(min(self.shape[2],rect[0]+1.25*rect[2])), int(min(self.shape[1],rect[1]+1.25*rect[3])))
                sub_face_index = [p1[1],min(p2[1],self.shape[1]),p1[0],min(p2[0], self.shape[2])]
                sub_face = self.frames[j_start][sub_face_index[0]:sub_face_index[1],sub_face_index[2]:sub_face_index[3]]
                sub_face = cv.GaussianBlur(sub_face, (171, 171), 60)
                for j in range(j_start, j_end):
                    if self.shape[0] <= j:
                        break
                    cv.rectangle(self.frames[j], p1, p2, (255, 255, 0), 4)
                    self.frames[j][sub_face_index[0]:sub_face_index[1],sub_face_index[2]:sub_face_index[3]] = sub_face
    
    # This function plays a video given an array of frames
    def play_vid(self, frames):
        print("Len Frames: ", len(frames))
        self.vid.set(cv.CAP_PROP_FPS, self.fps)
        cv.imshow("frame", frames[0])
        cv.waitKey(0)
        for frame in frames :
            cv.imshow('frame', frame)
            if cv.waitKey(4) & 0xFF == ord('q'):
                break
            
    # This function saves its own frames (WIP)
    def save_vid(self, save_path):
        save_path_obj = Path(save_path)
        vid_path_obj = Path(self.vid_dir)
        result = cv.VideoWriter(save_path,  
                         cv.VideoWriter_fourcc(*'H264'), 
                         self.fps, (self.shape[2], self.shape[1]))
        print("setting writer fps; ", result.get(cv.CAP_PROP_FPS), " to ", self.fps)
        for frame in self.frames:
            result.write(frame)
        result.release()
        # Get audio
        out_full_video = str(save_path_obj.parent.joinpath(str(save_path_obj.stem)+"full"+save_path_obj.suffix))
        out_audio = str(save_path_obj.parent.joinpath(str(save_path_obj.stem)+ ".wav"))
        print("out_full_video: ", out_full_video)
        print("out_audio: ", out_audio)
        FNULL = open(os.devnull, 'w')
        command_make_audio = ["ffmpeg", "-y", "-i", vid_path_obj, "-vn", out_audio]
        command_make_combined = ["ffmpeg", "-y", "-i", save_path, "-i", out_audio, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", out_full_video]
        command_rename = ["mv", out_full_video, save_path]
        subprocess.run(command_make_audio, stdout=FNULL, stderr=subprocess.STDOUT)
        subprocess.run(command_make_combined, stdout=FNULL, stderr=subprocess.STDOUT)
        os.remove(save_path)
        os.remove(save_path[:-4]+".wav")
        subprocess.run(command_rename, stdout=FNULL, stderr=subprocess.STDOUT)
        
# Driver function
def main():
    vid_dir = os.getcwd() + "/static/assets/test_data/vids/vids_test_load_func/test1.mp4"
    anon = Anon(vid_dir)
    t1 = time.time()
    print("TIME after construction: ", t1-t0)
    anon.anon_static()
    anon.save_vid(os.getcwd() + "/static/assets/test_data/vids/processed/testsave1.mp4")
    t6 = time.time()
    print("TIME (TOTAL) after saving: ", t6-t0)
    # anon.play_vid(anon.frames)
    
    print("Shape frames:", np.shape(anon.frames))
    print("One frame shape: ", np.shape(anon.frames[0]))

# Driver function used for testing and debugging
def main_test():
    vid_dir = os.getcwd() + "/static/assets/test_data/vids/vids_test_load_func/test2.mp4"
    path_dir = Path(vid_dir)
    # Downsample video file using ffmpeg
    command_get_frames = "ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 " + vid_dir
    command_get_shape = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 " + vid_dir
    command_rename = ["mv", os.path.join(path_dir.parent, "lowres"+path_dir.name), vid_dir]

    num_frames = subprocess.run(command_get_frames.split(), stdout=subprocess.PIPE).stdout.decode("utf-8")
    print("Num frames: ", num_frames)
    width, height = subprocess.run(command_get_shape.split(), stdout=subprocess.PIPE).stdout.decode("utf-8").split("x")
    print("pre Width height: ", width, ", ", height)
    width, height = (int(width), int(height))
    if width > 1500 or height > 1500:
        print("Supposedly adjusting said properties")
        scale = 1500/max(width, height)
        width = int(width * scale)
        height = int(height * scale)
        
        command_lowres = "ffmpeg -i " + vid_dir + " -s " + str(width) + "x" + str(height) + " -vcodec mpeg4 -q:v 20 -acodec copy " + os.path.join(path_dir.parent, "lowres"+path_dir.name)
        subprocess.run(command_lowres.split())
        # os.remove(vid_dir)
        # subprocess.run(command_rename, stdout=FNULL, stderr=subprocess.STDOUT)
    print("Width height: ", width, ", ", height)
    vid = cv.VideoCapture(vid_dir)
    
if __name__ == "__main__":
    main()
