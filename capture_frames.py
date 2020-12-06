import time
from datetime import datetime
import cv2
import numpy as np
import json
from picamera import PiCamera
from stereo_camera import *

# Camera settimgs
camera_width = 1280
camera_height = 480

# Scaling settings
scale_ratio = 0.5
hflip = False

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(camera_width, camera_height)
camera.framerate = 20
camera.hflip = hflip

# Initialize interface windows
cv2.namedWindow("Frame")
cv2.moveWindow("Frame", 50,100)

# capture frames from the camera
frame_width = int (camera_width * scale_ratio)
frame_height = int (camera_height * scale_ratio)
frame_buffer = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

# Odometry capture
odometry_sample_counter = 0
odometry_sample_interval = 750
odometry_frames_dir = './odometry_data/frames/'
odometry_last_sample_time = time.time()*1000

for frame in camera.capture_continuous(frame_buffer, format="bgra", use_video_port=True, resize=(frame_width, frame_height)):        
    cv2.imshow("Frame", frame)    
    
    if (time.time()*1000-odometry_last_sample_time > odometry_sample_interval):
        odometry_sample_counter += 1
        odometry_last_sample_time = time.time()*1000
        zeros = "0" * (5 - len(str(odometry_sample_counter)))
        frame_filename = "{0}/frame_{1}{2}.png".format(odometry_frames_dir, zeros, str(odometry_sample_counter))
        cv2.imwrite(frame_filename, frame)
               
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();




