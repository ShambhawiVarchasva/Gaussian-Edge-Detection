#Laplacian of Gaussian
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

def apply_filter(image,kernel_2D):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel_2D.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel_2D * padded_image[row:row + kernel_row, col:col + kernel_col])
            output[row, col] /= kernel_2D.shape[0] * kernel_2D.shape[1]

 
    return output


# Function for calculating the laplacian of the gaussian at a given point and with a given variance
def l_o_g(x, y, sigma):
	nom = ( (y**2)+(x**2)-2*(sigma**2) )
	denom = ( (2*math.pi*(sigma**6) ))
	expo = math.exp( -((x**2)+(y**2))/(2*(sigma**2)) )
	return nom*expo/denom

def create_filter(sigma, size = 7):
    w = math.ceil(float(size)*float(sigma))
    if(w%2 == 0):
        w = w + 1
    l_o_g_mask = []
    w_range = int(math.floor(w/2))
    print(w_range)
    for i in range(-w_range, w_range+1):
        print(i)
        for j in range(-w_range, w_range+1):
            print(j)
            l_o_g_mask.append(l_o_g(i,j,sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w,w)
    return l_o_g_mask

# Convolute the mask with the image. May only work for masks of odd dimensions
def convolve(image, mask):
	width = image.shape[1]
	height = image.shape[0]
	w_range = int(math.floor(mask.shape[0]/2))

	res_image = np.zeros((height, width))

	# Iterate over every pixel that can be covered by the mask
	for i in range(w_range,width-w_range):
		for j in range(w_range,height-w_range):
			# Then convolute with the mask 
			for k in range(-w_range,w_range):
				for h in range(-w_range,w_range):
					res_image[j, i] += mask[w_range+h,w_range+k]*image[j+h,i+k]
	return res_image

# Find the zero crossing in the l_o_g image
def z_c_test(img):
        (M,N) = img.shape
        #detect zero crossing by checking values across 8-neighbors on a 3x3 grid
        temp = np.zeros((M+2,N+2))
        temp[1:-1,1:-1] = img
        img = np.zeros((M,N))
        for i in range(1,M+1):
            for j in range(1,N+1):
                if temp[i,j]<0:
                    for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                            if temp[i+x,j+y]>0:
                                img[i-1,j-1] = 1
        return img

img = cv2.imread('house.jpg', 0) 
kernel1=np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]])
kernel2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
img=apply_filter(img,kernel1)
for sigma in range(1,4):
    kernel1 = create_filter(sigma)
    convolved_image1 = convolve(img, kernel1)
    #convolved_image2 = convolve(img, kernel2)
    cv2.imshow('Smoothened Image Sigma '+str(sigma), convolved_image1) 
    cv2.waitKey(0)
    #cv2.imshow('Smoothened Image2', convolved_image2)
    log_image1 = z_c_test(convolved_image1)
    #log_image2 = z_c_test(convolved_image2)
    cv2.imshow('LOG Image Sigma '+str(sigma), log_image1) 
    cv2.waitKey(0)

#cv2.imshow('LOG Image2', log_image2)
