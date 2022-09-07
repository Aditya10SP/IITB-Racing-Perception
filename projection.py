# from tkinter import W
import cv2
import numpy as np

def propagate(left_pts, left_img, right_img, radius = 2, draw = 0):
	sift = cv2.SIFT_create()
	right_pts = []
	for pt in left_pts:
		x,y = pt
		left_kpt = cv2.KeyPoint(x,y,radius)
		right_kpts = []
		length = right_img.shape[0]
		for i in range(1, length, radius):
			temp = cv2.KeyPoint(i, y, radius)
			right_kpts.append(temp)

		kp, rdesc = sift.compute(right_img, right_kpts)
		kp, ldesc = sift.compute(left_img, [left_kpt])
		bf = cv2.BFMatcher()
		matches = bf.match(ldesc, rdesc)
		matches = sorted(matches, key=lambda a: a.distance)

		for match in matches:
			idx = match.trainIdx
			(xr,yr) = right_kpts[idx].pt
			right_pts.append((int(xr),int(yr)))

	if draw == 1:
		imgout = np.concatenate((left_img, right_img), axis=1)
		for i in range(len(right_pts)):
			color = tuple(np.random.randint(0,255,3))
			color = (int(color[0]), int(color[1]), int(color[2]))
			imgout = cv2.circle(imgout, left_pts[i], 3, color, 1)
			right = (int(right_pts[i][0]+left_img.shape[0]), int(right_pts[i][1]))
			imgout = cv2.circle(imgout, right, 3, color, 1)
			imgout = cv2.line(imgout, left_pts[i], right, color, 1)

		cv2.imshow("Matched points", imgout)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return right_pts

def get_bbox_from_kpts(kpts, img = None, draw = 0):
	for i in range(len(kpts)):
		cone_kpts= kpts[i]
		cone_kpts = sorted(cone_kpts, key = lambda pts: pts[0])
		w = cone_kpts[-1][0] - cone_kpts[0][0] 
		h = cone_kpts[0][1] - cone_kpts[3][1] 		# change indices according to the sequence
		(x,y) = cone_kpts[0]
		y -= h
		x-=w
		color = tuple(np.random.randint(0, 255, 3))
		color = (int(color[0]), int(color[1]), int(color[2]))
		out_img = cv2.rectangle(img, (x,y), (x+w,y+h), color, 3) 
	cv2.imshow("Matched points kpt", out_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return (x,y,w,h)

def draw_bbox(bb, img): 
	xr,yr,w,h = bb   
	xr = int(xr)
	yr = int(yr)
	w = int(w)
	h = int(h)
	color = tuple(np.random.randint(0, 255, 3))
	color = (int(color[0]), int(color[1]), int(color[2]))
	out_img = cv2.rectangle(img, (xr,yr), (xr+w,yr+h), color, 3)
	cv2.imshow("Matched points bb", out_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return (xr,yr,w,h)


if __name__ == "__main__":
	left_image = cv2.imread('../stereo_image/left_image.jpeg')
	right_image = cv2.imread('../stereo_image/right_image.jpeg')

	left_pts = []
	pts = propagate(left_pts, left_image, right_image, draw=0)
	get_bbox_from_kpts(pts, right_image, 1)

