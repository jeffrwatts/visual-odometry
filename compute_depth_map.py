import json
from stereo_camera import *
import matplotlib.pyplot as plt

hflip = False
image_dir = './calibrate_images'

if (hflip):
    image_dir = image_dir + '_hflip'
    
image_filename = image_dir + '/test.jpg'
config_json_filename = image_dir + '/sbm_config.json'

calibration_data = np.load(image_dir + '/calibration_data.npz')
        
hflip = calibration_data['hflip']
left_map_1 = calibration_data['left_map_1']
left_map_2 = calibration_data['left_map_2']
right_map_1 = calibration_data['right_map_1']
right_map_2 = calibration_data['right_map_2']
Q = calibration_data['Q']
print(Q)

with open(config_json_filename) as sbm_config_file:
    sbm_config = json.load(sbm_config_file)
    
sbm = create_SBM(sbm_config)

image_pair = cv2.imread(image_filename, cv2.IMREAD_COLOR)

_3dImage, disparity, _, _ = compute_3dImage(sbm, image_pair, left_map_1, left_map_2, right_map_1, right_map_2, Q, hflip)

local_max = disparity.max()
local_min = disparity.min()
disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
plt.imshow(disparity_visual, aspect='equal', cmap='jet')
plt.show()

left_camera_matrix = calibration_data['left_camera_matrix']
T = calibration_data['T']

f = left_camera_matrix[0,0]
b = abs(T[0])
print(f)
print(b)
safe_disparity = disparity
safe_disparity[safe_disparity == 0] = 0.1
safe_disparity[safe_disparity == -1] = 0.1

# Initialize the depth map to match the size of the disparity map
depth_map = np.ones(safe_disparity.shape, np.single)

# Calculate the depths 
depth_map[:] = f * b / safe_disparity[:]

print("Ukulele at 0.75m (_3dImage)")
print(_3dImage[140:145,125:130,2])
print("Ukulele at 0.75m (depth_map)")
print(depth_map[140:145,125:130])

print("Guitar at 1.75m (_3dImage)")
print(_3dImage[125:130,200:205,2])
print("Guitar at 0.75m (depth_map)")
print(depth_map[125:130,200:205])