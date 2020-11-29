import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_image(image_pair):
    height = image_pair.shape[0]
    width_pair = image_pair.shape[1]
    width = int(width_pair/2)

    image_left = image_pair[0:height, 0:width]
    image_right = image_pair[0:height, width:width_pair]
    
    return image_left, image_right, (width, height)

def get_corners (image_search, checker_def, scale_ratio = 1):
    columns = checker_def[0]
    rows = checker_def[1]
    square_size = checker_def[2]
    
    object_point = np.zeros((columns*rows, 3), dtype=np.float32)
    object_point[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)
    object_point *= square_size

    # Arrays to store object points and image points from all the images.
    object_points = [] # 3d point in real world space
    left_image_points = [] # 2d points in left camera image plane.
    right_image_points = [] # 2d points in right camera image plane.

    images = glob.glob(image_search)

    for image_filename in images:
        image_pair = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        image_pair = cv2.cvtColor(image_pair,cv2.COLOR_BGR2GRAY)
        image_left, image_right, (width, height) = split_image(image_pair)

        # Find the chess board corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
        retL, cornersL = cv2.findChessboardCorners(image_left, (columns, rows), flags)
        retR, cornersR = cv2.findChessboardCorners(image_right, (columns, rows), flags)

        if ((retL == True) and (retR == True)):
            cornersL = cornersL*scale_ratio 
            cornersR = cornersR*scale_ratio
            
            scaled_width = int(width*scale_ratio)
            scaled_height = int(height*scale_ratio)

            resized_image_left = cv2.resize (image_left, (scaled_width,scaled_height), interpolation = cv2.INTER_AREA)
            resized_image_right = cv2.resize (image_right, (scaled_width,scaled_height), interpolation = cv2.INTER_AREA)
                     
            object_points.append(object_point)           
            
            subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            cornersL = cv2.cornerSubPix(resized_image_left, cornersL, (3,3), (-1,-1), subpix_criteria)
            left_image_points.append(cornersL)
            cornersR = cv2.cornerSubPix(resized_image_right, cornersR, (3,3), (-1,-1), subpix_criteria)
            right_image_points.append(cornersR)
            
    return object_points, left_image_points, right_image_points, (scaled_width,scaled_height)

def calibrate_cameras(object_points, left_image_points, right_image_points, image_dims):
    calibration_data = {}

     # Calibrate Left Camera
    (rms_left, left_camera_matrix, 
     left_dist_coeffs, _, _) = cv2.calibrateCamera(objectPoints=object_points,
                                                   imagePoints=left_image_points,
                                                   imageSize=image_dims,
                                                   cameraMatrix=None,
                                                   distCoeffs=None)
    calibration_data['rms_left'] = rms_left
    calibration_data['left_camera_matrix']  = left_camera_matrix
    calibration_data['left_dist_coeffs']  = left_dist_coeffs
                
    # Calibrate Right Camera
    (rms_right, right_camera_matrix, 
     right_dist_coeffs, _, _) = cv2.calibrateCamera(objectPoints=object_points,
                                                   imagePoints=right_image_points,
                                                   imageSize=image_dims,
                                                   cameraMatrix=None,
                                                   distCoeffs=None)
    calibration_data['rms_right'] = rms_right
    calibration_data['right_camera_matrix']  = right_camera_matrix
    calibration_data['right_dist_coeffs']  = right_dist_coeffs 

    # Stereo Calibrate
    (rms_stereo, _, _, _, _,
     R, T, E, F) = cv2.stereoCalibrate(objectPoints=object_points,
                                       imagePoints1=left_image_points,
                                       imagePoints2=right_image_points,
                                       cameraMatrix1=left_camera_matrix,
                                       distCoeffs1=left_dist_coeffs,
                                       cameraMatrix2=right_camera_matrix,
                                       distCoeffs2=right_dist_coeffs,
                                       imageSize=image_dims)
    calibration_data['rms_stereo'] = rms_stereo
    calibration_data['R']  = R
    calibration_data['T']  = T 
    calibration_data['E']  = E 
    calibration_data['F']  = F 

     # Compute parameters to rectify each camera.
    (rect_trans_left, rect_trans_right, 
     proj_matrix_left, proj_matrix_right, 
     Q, boxes_left, boxes_right) = cv2.stereoRectify(cameraMatrix1=left_camera_matrix,
                                                     distCoeffs1=left_dist_coeffs,
                                                     cameraMatrix2=right_camera_matrix,
                                                     distCoeffs2=right_dist_coeffs,
                                                     imageSize=image_dims,
                                                     R=R,
                                                     T=T)
     
    # Compute the undistort maps
    left_map_1, left_map_2 = cv2.initUndistortRectifyMap(cameraMatrix=left_camera_matrix,
                                                        distCoeffs=left_dist_coeffs,
                                                        R=rect_trans_left,
                                                        newCameraMatrix=proj_matrix_left,
                                                        size=image_dims,
                                                        m1type=cv2.CV_32FC1)

    right_map_1, right_map_2 = cv2.initUndistortRectifyMap(cameraMatrix=right_camera_matrix,
                                                    distCoeffs=right_dist_coeffs,
                                                    R=rect_trans_right,
                                                    newCameraMatrix=proj_matrix_right,
                                                    size=image_dims,
                                                    m1type=cv2.CV_32FC1)

    calibration_data['left_map_1']  = left_map_1
    calibration_data['left_map_2']  = left_map_2 
    calibration_data['right_map_1']  = right_map_1 
    calibration_data['right_map_2']  = right_map_2     
    calibration_data['Q']  = Q 
    return calibration_data

def save_calibration_data (calibration_data, calibration_file):
    np.savez(calibration_file, 
    rms_left = calibration_data['rms_left'], 
    left_camera_matrix = calibration_data['left_camera_matrix'],
    left_dist_coeffs = calibration_data['left_dist_coeffs'],
    rms_right = calibration_data['rms_right'],
    right_camera_matrix = calibration_data['right_camera_matrix'],
    right_dist_coeffs = calibration_data['right_dist_coeffs'],
    rms_stereo = calibration_data['rms_stereo'],
    R = calibration_data['R'],
    T = calibration_data['T'],
    E = calibration_data['E'],
    F = calibration_data['F'],
    Q = calibration_data['Q'],
    left_map_1 = calibration_data['left_map_1'],
    left_map_2 = calibration_data['left_map_2'],
    right_map_1 = calibration_data['right_map_1'],
    right_map_2 = calibration_data['right_map_2'])

def create_SBM (sbm_config):
    block_size = 15

    sbm = cv2.StereoBM_create(numDisparities=sbm_config['numberOfDisparities'], blockSize=block_size)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(sbm_config['preFilterSize'])
    sbm.setPreFilterCap(sbm_config['preFilterCap'])
    sbm.setMinDisparity(sbm_config['minDisparity'])
    sbm.setTextureThreshold(sbm_config['textureThreshold'])
    sbm.setUniquenessRatio(sbm_config['uniquenessRatio'])
    sbm.setSpeckleRange(sbm_config['speckleRange'])
    sbm.setSpeckleWindowSize(sbm_config['speckleWindowSize'])

    return sbm

def compute_3dImage(sbm, image_pair, left_map_1, left_map_2, right_map_1, right_map_2, Q): 
    image_left, image_right, _ = split_image(image_pair)
    
    gray_left = cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)

    rectified_image_left = cv2.remap(gray_left, left_map_1, left_map_2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified_image_right = cv2.remap(gray_right, right_map_1, right_map_2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    disparity = sbm.compute(rectified_image_left, rectified_image_right).astype(np.float32) / 16.0
    _3dImage = cv2.reprojectImageTo3D(disparity, Q)

    return _3dImage, disparity, image_left, image_right


