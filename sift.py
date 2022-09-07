import torch
import numpy as np
import cv2


class Features:

	def __init__(self):
		self.sift = cv2.SIFT_create()
		self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

	def feature_detect(self, left_boxes, right_boxes, left_img, right_img, draw=0):
		"""
		Extract feature points from the left and right images for all the provided bounding boxes using SIFT algorithm

		:param right_img: Image from right camera
		:param left_img: Image from left camera
		:param left_boxes: Bounding Box params for left image
		:param right_boxes:	Bounding Box params propagated on right image
		:param draw: Boolean variable. If 1, draws the feature points and prints the image on the screen.
		:return: Feature keypoints and descriptors for left and right image

		"""
		self.left_image = left_img
		self.right_image = right_img

		# Left Boxes
		left_keypoints = []
		left_descriptors = []
		for left_box in left_boxes:
			for cone in left_box:
				# x, y, w, h = left_box
				cls, xywh, conf = cone
				x, y, w, h = xywh
				lx = left_image.shape[1]  # Width
				ly = left_image.shape[0]  # Height
				x1 = int((x - w / 2) * lx)
				y1 = int((y - h / 2) * ly)
				x2 = int((x + w / 2) * lx)
				y2 = int((y + h / 2) * ly)
				mask = np.zeros(self.left_image.shape[:2], dtype = 'uint8')
				mask[y1:y2,x1:x2] = np.ones(mask[y1:y2,x1:x2].shape,dtype=np.uint8)*255

				keypoints, descriptors = self.sift.detectAndCompute(self.left_image, mask=mask)
				left_keypoints.append(keypoints)
				left_descriptors.append(descriptors)

		# Right Box
		right_keypoints = []
		right_descriptors = []
		for right_box in right_boxes:
			for cone in right_box:
				# x, y, w, h = right_box
				cls, xywh, conf = cone
				x, y, w, h = xywh
				rx = right_image.shape[1]
				ry = right_image.shape[0]
				x1 = int((x - w / 2) * rx)
				y1 = int((y - h / 2) * ry)
				x2 = int((x + w / 2) * rx)
				y2 = int((y + h / 2) * ry)
				mask = np.zeros(self.right_image.shape[:2], dtype='uint8')
				mask[y1:y2, x1:x2] = np.ones(mask[y1:y2, x1:x2].shape, dtype=np.uint8) * 255

				keypoints, descriptors = self.sift.detectAndCompute(self.right_image, mask=mask)
				right_keypoints.append(keypoints)
				right_descriptors.append(descriptors)

		if draw == 1:
			sift_image = self.left_image
			for points in left_keypoints:
				sift_image = cv2.drawKeypoints(sift_image, points, outImage=None)
			sift_image = cv2.resize(sift_image,(1280,720))
			cv2.imshow("image", sift_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		if draw == 1:
			sift_image = self.right_image
			for points in right_keypoints:
				sift_image = cv2.drawKeypoints(sift_image, points, outImage=None)
			sift_image = cv2.resize(sift_image,(1280,720))
			cv2.imshow("image", sift_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		features = {"keypoints": [left_keypoints, right_keypoints],
					"descriptors": [left_descriptors, right_descriptors]}

		return features

	def feature_matching(self, features, draw=1):
		"""
		Match feature points using Lowe's ratio test and return matched keypoints in left and right image for each bbox

		:param features: Dict containing left and right feature keypoints and descriptors for each box in frame.
		:param draw: Boolean variable. If 1, draws the matched feature points and prints the image on the screen.
		:return keypoints: keypoints["left"][i] will store coordinates of all the feature points for box i in left frame (as floats)
		"""

		left_descriptor = features["descriptors"][0]
		right_descriptor = features["descriptors"][1]
		left_keypoints = features["keypoints"][0]
		right_keypoints = features["keypoints"][1]

		all_matches = []		# stores all cv2::DMatch (from left to right) objects for each box in frame

		for i in range(len(left_keypoints)):
			knn_matches = self.matcher.knnMatch(left_descriptor[i], right_descriptor[i], 2)
			ratio_thresh = 0.7
			good_matches = []		# matches for box i
			for m, n in knn_matches:
				if m.distance < ratio_thresh * n.distance:
					good_matches.append(m)
			all_matches.append(good_matches)

		# Get KeyPoint Coordinates
		keypoints = {"left": [], "right": []}
		for i in range(len(all_matches)):
			box_matches = all_matches[i]
			box_kp_l = left_keypoints[i]
			box_kp_r = right_keypoints[i]
			pts_l = []
			pts_r = []
			for match in box_matches:
				point_l = tuple(map(int,box_kp_l[match.queryIdx].pt))
				point_r = tuple(map(int,box_kp_r[match.trainIdx].pt))
				pts_l.append(point_l)
				pts_r.append(point_r)

			# keypoints["left"][i] will store coordinates of all the feature points for box i in left frame (as floats)
			keypoints["left"].append(pts_l)
			keypoints["right"].append(pts_r)

		# Visualization and Debugging
		if draw == 1:
			for i in range(len(all_matches)):
				img_matches = np.empty((max(self.left_image.shape[0], self.right_image.shape[0]),
										self.left_image.shape[1] + self.right_image.shape[1], 3), dtype=np.uint8)
				cv2.drawMatches(self.left_image, left_keypoints[i], self.right_image, right_keypoints[i], all_matches[i],
								img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
				img_matches = cv2.resize(img_matches,(2000,720))
				cv2.imshow('Good Matches', img_matches)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

		return keypoints

if __name__ == '__main__':
	left_image = cv2.imread("E:\Racing\stereo-depth-estimation\stereo_image\yolov5fsdstest\content\yolov5\\runs\detect\exp2\left1.jpg")
	right_image = cv2.imread("E:\Racing\stereo-depth-estimation\stereo_image\yolov5fsdstest\content\yolov5\\runs\detect\exp2\\right1.jpg")
	# left_image = cv2.resize(left_image,(80,80))
	# right_image = cv2.resize(right_image,(80,80))
	xyz = Features()
	# left_image = cv2.resize(left_image,(320,320))

	# lx = left_image.shape[1]  # Width
	# ly = left_image.shape[0]  # Height
	# rx = right_image.shape[1]
	# ry = right_image.shape[0]
	right_boxes = [[0.27225, 0.725556, 0.0775, 0.265556],[0.3865, 0.436389, 0.031, 0.132778],[0.509125, 0.416111, 0.03275, 0.125556]]
	left_boxes = [[0.14575, 0.626111, 0.072, 0.272222],[0.367125, 0.368333, 0.03425, 0.135556],[0.502, 0.354444, 0.031, 0.108889]]
	# x,y,w,h = left_boxes
	# x1 = int(x-w/2)
	# y1 = int(y-h/2)
	# x2 = int(x+w/2)
	# y2 = int(y+h/2)
	# left_image = cv2.rectangle(left_image,(x1,y1),(x2,y2),(255,0,0),-1)
	# left_image = cv2.resize(left_image,(1280,720))
	feat = xyz.feature_detect(left_boxes, right_boxes, left_image, right_image, draw=0)
	kpts = xyz.feature_matching(feat, draw=0)
	print(kpts)
	# x, y, w, h = left_boxes
	# mask = np.zeros(left_image.shape[:2], dtype='uint8')
	# mask[y1:y2,x1:x2] = np.ones(mask[y1:y2,x1:x2].shape,dtype=np.uint8)*255
	# left_image = cv2.bitwise_and(left_image,left_image, mask=mask)
	# left_image = cv2.resize(left_image,(1280,720))
	# cv2.imshow("masked",left_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


