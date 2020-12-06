import time
import cv2
import json
import numpy as np
from picamera import PiCamera
from stereo_camera import *

# Camera settimgs
camera_width = 1280
camera_height = 480

# Scaling settings
scale_ratio = 0.5
hflip = False

image_dir = './calibrate_images'

if (hflip):
    image_dir = image_dir + '_hflip'
    
conifgJsonFile = image_dir + '/sbm_config.json'

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(camera_width, camera_height)
camera.framerate = 20
camera.hflip = hflip

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("Disparity")
cv2.moveWindow("Disparity", 450,100)

calibration_data = np.load(image_dir + '/calibration_data.npz')
   
hflip = calibration_data['hflip']
left_map_1 = calibration_data['left_map_1']
left_map_2 = calibration_data['left_map_2']
right_map_1 = calibration_data['right_map_1']
right_map_2 = calibration_data['right_map_2']
Q = calibration_data['Q']
print(Q)

with open(conifgJsonFile) as sbm_config_file:
    sbm_config = json.load(sbm_config_file)
    
sbm = create_SBM(sbm_config)

# capture frames from the camera
frame_width = int (camera_width * scale_ratio)
frame_height = int (camera_height * scale_ratio)
frame_buffer = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

# Display Distance and Disparity Map
display_distance = True
disp_max = -100000
disp_min = 100000

for frame in camera.capture_continuous(frame_buffer, format="bgra", use_video_port=True, resize=(frame_width, frame_height)):
 
    _3dImage, disparity, image_left, _ = compute_3dImage(sbm, frame, left_map_1, left_map_2, right_map_1, right_map_2, Q, hflip)

    depth_map = _3dImage[:,:,2]
    depth_map[depth_map < 0] = 0.0
    depth_map[depth_map == np.inf] = 0.0
    
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
    
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();



