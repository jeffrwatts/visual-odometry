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

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(camera_width, camera_height)
camera.framerate = 20
camera.hflip = True

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("Disparity")
cv2.moveWindow("Disparity", 450,100)

calibration_data = np.load('calibration_data.npz')
        
left_map_1 = calibration_data['left_map_1']
left_map_2 = calibration_data['left_map_2']
right_map_1 = calibration_data['right_map_1']
right_map_2 = calibration_data['right_map_2']
Q = calibration_data['Q']
print(Q)

with open('sbm_config.json') as sbm_config_file:
    sbm_config = json.load(sbm_config_file)
    
sbm = create_SBM(sbm_config)

# capture frames from the camera
frame_width = int (camera_width * scale_ratio)
frame_height = int (camera_height * scale_ratio)
frame_buffer = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

# Display Distance and Disparity Map
display_distance = False
disp_max = -100000
disp_min = 100000

# Odometry capture
odometry_sample_capture = True
odometry_samples = 51
odometry_sample_counter = 0
odometry_sample_interval = 1000
odometry_image_dir = './odometry_data/images/'
odometry_depth_dir = './odometry_data/depths/'
odometry_last_sample_time = time.time()*1000


for frame in camera.capture_continuous(frame_buffer, format="bgra", use_video_port=True, resize=(frame_width, frame_height)):
    
    _3dImage, disparity, image_left, _ = compute_3dImage(sbm, frame, left_map_1, left_map_2, right_map_1, right_map_2, Q)

    depth_map = _3dImage[:,:,2]
    depth_map[depth_map < 0] = 0.0
    depth_map[depth_map == np.inf] = 0.0
    
    if (display_distance):
        depth_map = depth_map[115:125, 155:165]
        n_valid_distance = np.count_nonzero(depth_map)
        if (n_valid_distance != 0):
            distance = str(int(np.sum(depth_map)/n_valid_distance))
        else:
            distance = "invalid"

        image_left = cv2.rectangle(image_left, (155,115), (165,125), (0, 255, 0), thickness=2)
    
        cv2.putText(image_left, str(distance), (100,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        disp_max = max(disparity.max(),disp_max)
        disp_min = min(disparity.min(),disp_min)
        local_max = disp_max
        local_min = disp_min
        disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        disparity_color = cv2.rectangle(disparity_color, (155,115), (165,125), (0, 255, 0), thickness=2)
        cv2.imshow("Disparity", disparity_color)
        
    cv2.imshow("Image", image_left)    
    
    if (odometry_sample_capture and (time.time()*1000-odometry_last_sample_time > odometry_sample_interval)):
        t1 = time.time()*1000
        odometry_sample_counter += 1
        odometry_last_sample_time = time.time()*1000
        zeros = "0" * (5 - len(str(odometry_sample_counter)))
        image_filename = "{0}/frame_{1}{2}.png".format(odometry_image_dir, zeros, str(odometry_sample_counter))
        depth_filename = "{0}/frame_{1}{2}.npy".format(odometry_depth_dir, zeros, str(odometry_sample_counter))
        cv2.imwrite(image_filename, image_left)
        np.save(depth_filename, depth_map)
        if (odometry_sample_counter > odometry_samples):
            quit();
               
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();



