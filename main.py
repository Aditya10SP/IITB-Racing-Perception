# import sys
# print(sys.path)

# python3 -m venv .stereoenv
# source .stereoenv/bin/activate

import cv2
import torch
import numpy as np
from object_detect.yolo import Yolo
from Keypoint_Detection.Keypoint import Keypoints
from sift import Features
from triangulator import Triangulation
from triangulator import DepthFinder
from projection import *
from view_utils import get_shape
from view_utils import get_kpt_matches
from view_utils import draw_propagate


"""
DEFINING IMAGE, WEIGHT PATHS

TODO: Implement call of video from webcam/camera
"""

yolo_weights = './object_detect/yolov5/weights/last.pt'
yolo_data = './object_detect/yolov5/models/data.yaml'
kpr_path = "./keypoint_detection/Weights/23_loss_0.38.pt"

left_img_path = "stereo_image/left_image.jpeg"
right_img_path = "stereo_image/right_image.jpeg"

left_image = cv2.imread(left_img_path)
right_image = cv2.imread(right_img_path)			# Images in format (H,W,C)
vis_left_img = cv2.imread(left_img_path)
vis_right_img = cv2.imread(right_img_path)			# Images in format (H,W,C)

gn = torch.tensor(left_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
height,width,_ = left_image.shape


"""
SETUP YOLO, REKTNET MODELS
CREATE FEATURE MATCHING, TRIANGULATION, PnP OBJECTS
SETUP CAMERA PROPERTIES

TODO: Get proper baseline/focus from FSDS
"""

print('\n-----------------------------------------------------------------------------')
camera_data = {'focus':100, 'baseline':10, 'pixel_size':100 }	

yolo_model = Yolo(weights=yolo_weights, data=yolo_data, imgsz=[1280,1280], device='cpu')
kpr_model = Keypoints(kpr_path)

feature_match = Features()
triangulator1 = DepthFinder(500000,120,0.002)
triangulator2 = Triangulation(camera_data["focus"],camera_data["baseline"],camera_data["pixel_size"])
# pnp_model = PnP(camera_data)


"""
BOUNDING BOX + KEYPOINT DETECTION
BOUNDING BOX PROPAGATION
SAVING INFERENCES

Key Variables -->
	
	> left_boxes: list of YOLO predicted bounding boxes for all cones in left image, SHAPE: (num_cones, 3), 3 for [class_cone, xywh_cone, pred_confidence]
	> right_boxes: list of propagated bounding boxes for all cones in right image, SHAPE: (num_cones, 3), 3 for [class_cone, xywh_cone, pred_confidence]
	> left_kpts: list of predicted keypoints on the left camera's image, SHAPE: (num_cones, 7, 2), 7x2 for 7 keypoints' xy coordinates.
	> right_kpts: list of propagated keypoints on the right camera's image, SHAPE: (num_cones, 7, 2), 7x2 for 7 keypoints' xy coordinates.

TODO: Propagation implementation with PnP instead of using SIFT matching.
"""

conf_thresh = 0.85
left_boxes = yolo_model.detect_all(source=left_img_path, conf_thres=conf_thresh)[0]
print('\n-----------------------------------------------------------------------------')

left_boxes_shape = get_shape(left_boxes)
print(f'{left_boxes_shape[0]} cones detected with confidence > {conf_thresh}!\n')
print('Shape of left_boxes: ', left_boxes_shape)
#print('Shape of left_boxes xywh: ', get_shape(left_boxes[0][1]), '\n')

left_kpts = []
right_kpts = []
right_boxes = []

for conebb in left_boxes:

	''' Get bounding box properties '''
	cls, xywh, conf = conebb
	x, y, w, h = xywh

	''' Convert BB outputs: "xywh": center , height, width --->  "xyxy": opposite corners '''
	x1 = int((x - w / 2) * width)
	y1 = int((y - h / 2) * height)
	x2 = int((x + w / 2) * width)
	y2 = int((y + h / 2) * height)
	
	''' Extract bounding box and get left(predicted) keypoints '''
	cone_img = left_image[y1:y2,x1:x2]
	cheight, cwidth, _ = cone_img.shape
	kpts = kpr_model.get_keypoints(cone_img)
	kpts = np.array(kpts * [[cwidth,cheight]])

	''' Get left(predicted) image keypoints and visualize on the complete image '''
	left_pts = []
	for pt in kpts:
		cvpt = (int(pt[0]+((x - w / 2) * width)), int(pt[1]+((y - h / 2) * height)))
		cv2.circle(vis_left_img, cvpt, 3, (0, 255, 0), -1)
		left_pts.append(list(cvpt))

	''' Get right(propagated) image keypoints and visualize on the complete image '''
	#right_pts=propagate(kpts, left_image, right_image, draw=0)
	#for pt in right_pts:
	#	cvpt = (int(pt[0]+((x - w / 2) * width)), int(pt[1]+((y - h / 2) * height)))
	#	cv2.circle(vis_right_img, cvpt, 3, (0, 255, 0), -1)

	''' Save keypoints for this pair of images '''
	#right_kpts.append(right_pts)
	left_kpts.append(left_pts)

	''' Get right image bounding boxes, show and save '''
	#right_bbox = get_bbox_from_kpts(right_pts, img = vis_right_img, draw = 0)
	# draw_bbox(right_bbox, vis_right_img)
	#right_conebb = [cls, right_bbox, conf]
	#right_boxes.append(right_conebb)

print('Shape of left_kpts: ', get_shape(left_kpts))
#print('Shape of right_kpts: ', get_shape(right_kpts))

print(f'Left image saved as "results_bbkpr/result_left": {cv2.imwrite("./results_bbkpr/result_left.png", vis_left_img)}')
print(f'Right image saved as "results_bbkpr/result_right": {cv2.imwrite("./results_bbkpr/result_right.png", vis_right_img)}')
print('-----------------------------------------------------------------------------\n')


"""
FEATURE MATCHING
TRIANGULATION

Key Variables -->
	
	> 

TODO: Restart coding of feature matching from scratch using helper functions defined in view_utils.py, sift.py
"""
all_lkpts = np.array(left_kpts)
right_new_kpts = get_kpt_matches(left_image, right_image, left_kpts)
all_rkpts = np.array(get_kpt_matches(left_image, right_image, left_kpts))
n = len(all_lkpts)
#for i in range(int(n)): 
#	tensor_left_points = torch.tensor(all_lkpts[i], dtype=torch.float32)	
#	tensor_right_points = torch.tensor(all_rkpts[i], dtype=torch.float32)
#	depths1 = triangulator1.find_depth(tensor_left_points, tensor_right_points)
#	depth1_in_meter = depths1.item()/1000
#	format_depth1 = "{:.2f}".format(depth1_in_meter)
#	print("depth1 of cone", " " , i+1 , " " , format_depth1)
#	depth1_left_img = cv2.putText(left_image, str(format_depth1), all_lkpts[i][0], cv2.FONT_HERSHEY_SIMPLEX, 
#								0.4, (0,255,0), 1, cv2.LINE_AA)
#print(cv2.imwrite("./depth_result_left.png",depth1_left_img))
#draw_propagate(all_lkpts, all_rkpts, left_image, right_image)


""" BOUNDING BOXES IN THE RIGHT IMAGE FROM RIGHT KEYPOINTS """
get_bbox_from_kpts(right_new_kpts,img=vis_right_img,draw=1)


quit()

