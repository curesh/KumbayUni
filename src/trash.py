import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

def main_bad():
    vid_dir = os.getcwd() + "/test_data/test2.mp4"
    
    vid = cv.VideoCapture(vid_dir)
    fps = vid.get(cv.CAP_PROP_FPS)
    print("FPS: ", fps)
    ret, frame = vid.read()
    
    cv.imshow("flip", frame)
    cv.waitKey(0)
    while(vid.isOpened()):
        
        ret, frame = vid.read()
        cv.imshow("frame", cv.flip(frame, 1))
        cv.waitKey(1)
    vid.release()
    cv.destroyAllWindows()

def main():
    img_dir = os.getcwd() + "/test_data/different_sized_faces.png"
    img = cv.imread(img_dir, 0)
    face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
    front_faces = face_cascade.detectMultiScale(img, 1.1, 4)
    print("front faces: ", front_faces)
    print("front faces shape: ", front_faces.shape)
    for (x,y,w,h) in front_faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.imshow("img", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
if __name__ == "__main__":
    main()