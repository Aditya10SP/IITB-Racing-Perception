import sys
import matplotlib.pyplot as plt
import tkinter
import cv2
import numpy as np
from view_utils import *
from matplotlib import pyplot as plt
import torch
from object_detect.yolo import Yolo
from Keypoint_Detection.Keypoint import Keypoints
from sift import Features
from triangulator import DepthFinder
from projection import *
import pickle
import time
import math

#TODO: Fix view_as_windows for very small bounding boxes, patch size exceeds bb size
yolo_weights = './object_detect/yolov5/weights/last.pt'
yolo_data = './object_detect/yolov5/models/data.yaml'
kpr_path = "./keypoint_detection/Weights/23_loss_0.38.pt"
conf_thresh = 0.8
yolo_model = Yolo(weights=yolo_weights, data=yolo_data, imgsz=[1280,1280], device='cpu')
kpr_model = Keypoints(kpr_path)
triangulator = DepthFinder()
left_img_path = "stereo_image/2left.png"        
right_img_path = "stereo_image/2right.png"

left_image = cv2.imread(left_img_path)
right_image = cv2.imread(right_img_path)

left_boxes = yolo_model.detect_all(source=left_img_path, conf_thres=conf_thresh)[0]
l_kpts = get_kp_from_bb(left_boxes, left_image, kpr_model)
# print(l_kpts)
r_kpts = get_kpt_matches(left_image, right_image, l_kpts, patch_width=15, disp_range=128, metric='mae')
depths = triangulator.find_depth(torch.tensor(l_kpts), torch.tensor(r_kpts))
depths = [str(ele) for ele in (np.round(np.array((depths)/1000), decimals=2))]
print(depths)

def bearing(depth,cone_centre, img_shape):
    focal_length=250
    centre=(img_shape[0]//2,img_shape[1]//2)
    distance=(centre[1]-cone_centre[0])
    theta=180*np.arctan(distance/focal_length)/np.pi
    range=float(depth)/(np.cos(theta*(np.pi)/180))  #2D Range
    cone_height = 0.325
    camera_height = 0.8 
    height_diff=camera_height-cone_height
    range_3d=np.sqrt((range**2)+(height_diff**2))
    return theta, range_3d

cone_centres = []
for i in range(len(depths)):    
    conec_x = 0
    conec_y = 0
    for j in range(7): 
        conec_x = conec_x + int(l_kpts[i][j][0])
        conec_y = conec_y + int(l_kpts[i][j][1])
    conec_x = conec_x//7 
    conec_y = conec_y//7
    cone_centres.append([conec_x,conec_y])

print("cone_centres",cone_centres)

thetas = []
ranges = []
for i in range(len(depths)):
    cone_centre = cone_centres[i]
    theta,range_3d = bearing(depths[i],cone_centre, left_image.shape)
    thetas.append(theta)
    ranges.append(range_3d)

print("thetas",thetas)
print("ranges",ranges)
draw_propagate(l_kpts, r_kpts, left_image, right_image, annots=depths)
# draw_propagate(l_kpts, r_kpts, left_image, right_image, line=True)



