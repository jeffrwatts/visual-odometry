from stereo_camera import *
import matplotlib.pyplot as plt

image_search = './calibrate_images/*.png'
checker_def = (9,6,40) # 9x6 checkerboard; 40mm square size.
scale_ratio = 0.5

object_points, left_image_points, right_image_points, scaled_image_dims = get_corners (image_search, checker_def, scale_ratio)

calibration_data = calibrate_cameras(object_points, left_image_points, right_image_points, scaled_image_dims)

rms_left = calibration_data['rms_left']
left_camera_matrix = calibration_data['left_camera_matrix']
left_dist_coeffs = calibration_data['left_dist_coeffs']

print("Left Calibration Result:")
print (rms_left)
print (left_camera_matrix)
print (left_dist_coeffs)

rms_right = calibration_data['rms_right']
right_camera_matrix = calibration_data['right_camera_matrix']
right_dist_coeffs = calibration_data['right_dist_coeffs']

print("Right Calibration Result:")
print (rms_right)
print (right_camera_matrix)
print (right_dist_coeffs)

rms_stereo = calibration_data['rms_stereo']
T = calibration_data['T']
Q = calibration_data['Q']

print("Stereo Calibration Result:")
print(rms_stereo)
print(T)
print(Q)

left_map_1 = calibration_data['left_map_1']
left_map_2 = calibration_data['left_map_2']
right_map_1 = calibration_data['right_map_1']
right_map_2 = calibration_data['right_map_2']

image_filename = 'test.jpg'

image_pair = cv2.imread(image_filename, cv2.IMREAD_COLOR)
image_pair = cv2.cvtColor(image_pair, cv2.COLOR_BGR2GRAY)
image_left, image_right, _ = split_image(image_pair)

rectified_image_left = cv2.remap(image_left, left_map_1, left_map_2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
rectified_image_right = cv2.remap(image_right, right_map_1, right_map_2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

_, image_cells = plt.subplots(1, 2, figsize=(10, 10))
image_cells[0].imshow(rectified_image_left)
image_cells[0].set_title('left image')
image_cells[1].imshow(rectified_image_right)
image_cells[1].set_title('right image')
plt.show()

save_calibration_data(calibration_data, 'calibration_data.npz')



