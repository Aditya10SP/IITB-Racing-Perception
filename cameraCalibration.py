import numpy as np
import cv2 as cv
import glob
import math
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)*1000
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), cv.CALIB_CB_ADAPTIVE_THRESH
                    + cv.CALIB_CB_FAST_CHECK +
                    cv.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw 
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#model_points = np.array([2.5,12,0])
#image_points = np.array([580,337])
image_points = np.array([
                            (410, 231),
(398, 276),
(431, 273),
(395, 310),
(439, 311),
(392, 352),
(446, 350) 
                        ], dtype="double")
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
(success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, mtx, dist)

print(rotation_vector)
print(translation_vector)
print(mtx)
# print(rvecs.shape)
#print(np.array(rvecs[0]))
# print(tvecs.shape)
#print(np.array(tvecs[0]))

rot, x =cv.Rodrigues(rotation_vector)
# trans=tvecs[0]
comb_matrix=np.concatenate((rot,translation_vector), axis=1)
# # print(comb_matrix)
temp_matrix= [[0,0,0,1]]
new_matrix=np.concatenate((comb_matrix,temp_matrix))
temp_1= [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0]]
new=np.matmul(np.matmul(mtx,temp_1),new_matrix)
temp_2=[[0,0,0,1]]
new_new=np.concatenate((new,temp_2))
inverse = np.linalg.inv(new_new)
new_inverse = np.delete(inverse, 3, 0)
array1 = image_points[0]
temp_3=[1]
array=np.concatenate((array1,temp_3))
point = np.matmul(array,new_inverse)
#print(point)
depth = (1/point[3])*np.sqrt(point[0]**2+point[1]**2+point[2]**2)
print(abs(depth))

