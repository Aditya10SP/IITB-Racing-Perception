import numpy as np

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

def cone_centre(l_kpts):
    cone_centres = []
    for i in range(len(l_kpts)):    
        conec_x = 0
        conec_y = 0
        for j in range(7): 
            conec_x = conec_x + int(l_kpts[i][j][0])
            conec_y = conec_y + int(l_kpts[i][j][1])
        conec_x = conec_x//7 
        conec_y = conec_y//7
        cone_centres.append([conec_x,conec_y])
    return cone_centres

def theta_range(depths, cone_centres, left_image):
    thetas = []
    ranges = []
    for i in range(len(depths)):
        cone_centre = cone_centres[i]
        theta,range_3d = bearing(depths[i],cone_centre, left_image.shape)
        thetas.append(theta)
        ranges.append(range_3d)    
    return thetas,range_3d

