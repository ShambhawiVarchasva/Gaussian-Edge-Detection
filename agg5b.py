from matplotlib import pyplot as plt
import numpy as np
import math
import cv2

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

img = cv2.imread('taj.jpg', 0) 

cv2.imshow("Original Image",img)
cv2.waitKey(0)
kernel2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
img=apply_filter(img,kernel2)
kernel1 = create_filter(1)
kernel2 = create_filter(3)
s1 = convolve(img, kernel1)
s2 = convolve(img, kernel2)
img = s1 - s2
cv2.imshow("DOG Image",img)
cv2.waitKey(0)
cv2.imshow("DOG Image",img)
cv2.waitKey(0)
