import cv2
import torch
import numpy as np
import glob
import math
from object_detect.yolo import Yolo
from Keypoint_Detection.Keypoint import Keypoints
from random import sample
from tqdm import tqdm

#TODO: change to call from camera
img_path = "mono_image/WIN_20220522_11_34_44_Pro.jpg"
yolo_weights = './object_detect/yolov5/weights/last.pt'
yolo_data = './object_detect/yolov5/models/data.yaml'
kpr_path = "./Keypoint_Detection/Weights/23_loss_0.38.pt"

image = cv2.imread(img_path)			# Images in format (H,W,C)
gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
height,width,_ = image.shape
vis_img = cv2.imread(img_path)

# Setup
camera_data = {'focus':100, 'baseline':10, 'pixel_size':100 }				# {focus, baseline, pixel_size, }
yolo_model = Yolo(weights=yolo_weights,
				  data=yolo_data,
				  imgsz=[1280,1280],
				  device='cpu')
kpr_model = Keypoints(kpr_path)

boxes = yolo_model.detect_all(source=img_path,
								   conf_thres=0.6)

img_kpts = []

all_key_pts_wrt_img = []
for img in boxes:
        for cone in img:
            cls, xywh, conf = cone
            x, y, w, h = xywh
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)
            cone_img = image[y1:y2,x1:x2]
            cheight, cwidth, _ = cone_img.shape
            kpts = kpr_model.get_keypoints(cone_img)
            kpts = kpts * [[cwidth,cheight]]
            kpts = np.array(kpts)
            img_kpts.append(kpts)
            print("------------------------------------------------------------------------------------")
            for pt in kpts:
                cvpt = (int(pt[0]+((x - w / 2) * width)), int(pt[1]+((y - h / 2) * height)))
                all_key_pts_wrt_img.append(np.array(cvpt))
                cv2.circle(vis_img, cvpt, 3, (0, 255, 0), -1)

#print(all_key_pts_wrt_img)
all_key_pts_wrt_img = (np.array(all_key_pts_wrt_img, dtype="double")).T
all_key_pts_wrt_img = all_key_pts_wrt_img.T
#print(img_kpts)
print(cv2.imwrite("./result_img.png",vis_img))
print("Images saved as: result_img.png")

#CAMERA CALIBRATION

n_iter = 100
for n_samples in np.arange(8,20):
    print(f'\nNumber of random chessboard images: {n_samples}')
    depths = []
    for i in range(n_iter):
        if i%10==0:
            print(f'{i}/100')
    # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)*1000
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('chessboard_images2/*.jpg')
        images = sample(images,n_samples)

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7,6), cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK +
                            cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
        # Draw 
                cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(1000)
        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # 3D model points.

    #theta = (180/3.14)*(math.atan(12/2.5))

        model_points = (25.4)*np.array([
                            (2.5, 12.0, 0.0),             
                            (((2/3)*2.5), 8.0, 0.0),        
                            (((4/3)*2.5), 8.0, 0.0),     
                            ((2.5/3), 4.0, 0.0),
                            (((5/3)*2.5), 4.0, 0.0),      
                            (0.0, 0.0, 0.0),    
                            (5.0, 0.0, 0.0)      
                        ])

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, all_key_pts_wrt_img, mtx, dist)
        if i==0:
            camera_matrix_iter = np.expand_dims(mtx,0)
        else:
            camera_matrix_iter = np.concatenate([camera_matrix_iter, np.expand_dims(mtx,0)], axis=0)
        # print(camera_matrix_iter.shape)
        depths.append(translation_vector[2])

        # print("camera matrix",i+1,"\n",mtx)
        # print("depth using camera matrix",i+1,"\n",translation_vector[2])
    depths = np.array(depths)
    print(f"Mean Depth = {depths.mean()}, Standard Deviation = {depths.std()}")
    print(f"Mean Cam Matrix = \n{np.mean(camera_matrix_iter, axis=0)}, \n\nStandard Deviation = \n{np.std(camera_matrix_iter, axis=0)}")

# # print(rvecs.shape)
# #print(np.array(rvecs[0]))
# # print(tvecs.shape)
# #print(np.array(tvecs[0]))

# rot, x =cv2.Rodrigues(rotation_vector)
# # trans=tvecs[0]
# comb_matrix=np.concatenate((rot,translation_vector), axis=1)
# # # print(comb_matrix)
# temp_matrix= [[0,0,0,1]]
# new_matrix=np.concatenate((comb_matrix,temp_matrix))
# temp_1= [[1,0,0,0],
#          [0,1,0,0],
#          [0,0,1,0]]
# new=np.matmul(np.matmul(mtx,temp_1),new_matrix)
# temp_2=[[0,0,0,1]]
# new_new=np.concatenate((new,temp_2))
# inverse = np.linalg.inv(new_new)
# new_inverse = np.delete(inverse, 3, 0)
# array1 = image_points[0]
# temp_3=[1]
# array=np.concatenate((array1,temp_3))
# point = np.matmul(array,new_inverse)
# #print(point)
# depth = (1/point[3])*np.sqrt(point[0]**2+point[1]**2+point[2]**2)
# print(abs(depth))
