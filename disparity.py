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
from bearings import *
import pickle
import time
import math
# import tensorrt
# print(tensorrt.__version__)
# assert tensorrt.Builder(tensorrt.Logger())

print(torch.cuda.is_available())

#TODO: Fix view_as_windows for very small bounding boxes, patch size exceeds bb size
yolo_weights = './object_detect/yolov5/weights/last.pt'
yolo_data = './object_detect/yolov5/models/data.yaml'
kpr_path = "./keypoint_detection/Weights/23_loss_0.38.pt"
conf_thresh = 0.8
yolo_model = Yolo(weights=yolo_weights, data=yolo_data, imgsz=[1280,1280], device='1')
kpr_model = Keypoints(kpr_path)
triangulator = DepthFinder()
left_img_path = "stereo_image/1left.png"        
right_img_path = "stereo_image/1right.png"

left_image = cv2.imread(left_img_path)
right_image = cv2.imread(right_img_path)
print(left_image.shape)
left_boxes = yolo_model.detect_all(source=left_img_path, conf_thres=conf_thresh)[0]
l_kpts = get_kp_from_bb(left_boxes, left_image, kpr_model)
print(len(l_kpts))
print("*********************************************************")

r_kpts = get_kpt_matches(left_image, right_image, l_kpts, patch_width=15, disp_range=16, metric='rmse')
print(r_kpts)
depths = triangulator.find_depth(torch.tensor(l_kpts), torch.tensor(r_kpts))
depths = [str(ele) for ele in (np.round(np.array((depths)/1000), decimals=2))]
print(depths)

cone_centres = cone_centre(l_kpts)
thetas,range_3d = theta_range(depths,cone_centres,left_image)
print("cone_centres",cone_centres)
print("thetas",thetas)
print("3d_ranges",range_3d)

# draw_propagate(l_kpts, r_kpts, left_image, right_image, annots=depths)
# draw_propagate(l_kpts, r_kpts, left_image, right_image, line=True)



