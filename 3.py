import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
 
def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
    
 
    print("Kernel Shape : {}".format(kernel.shape))
 
 
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
    print("Output Image size : {}".format(output.shape))
 
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()
 
    return output


import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
 
 
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
 
def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()
 
    return kernel_2D
 
 
def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
 
    image = cv2.imread(args["image"])
    output1 = np.zeros(image.shape)
    output1=gaussian_blur(image, 3, verbose=True)
    output2 = np.zeros(image.shape)
    output2=gaussian_blur(image, 5, verbose=True)
    output3 = np.zeros(image.shape)
    output3=gaussian_blur(image, 7, verbose=True)
    image_row, image_col = output1.shape
    output=np.zeros(image.shape)
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = output1[row,col]+output2[row,col]
    
    array1 = np.array(output, dtype=np.uint8)


    new_image1 = Image.fromarray(array1)
    new_image1.save('new_3_5.png') 
    img = cv2.imread('new_3_5.png') 


    cv2.imshow('new_3_5', img)
    cv2.waitKey(0)		 



    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = output2[row,col]+output3[row,col]

    array2 = np.array(output, dtype=np.uint8)


    new_image2 = Image.fromarray(array2)
    new_image2.save('new_5_7.png') 

    img = cv2.imread('new_5_7.png') 


    cv2.imshow('new_5_7', img)
    cv2.waitKey(0)	
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = output1[row,col]+output3[row,col]

    array3 = np.array(output, dtype=np.uint8)


    new_image2 = Image.fromarray(array3)
    new_image2.save('new_3_7.png')
    img = cv2.imread('new_3_7.png') 


    cv2.imshow('new_3_7', img)
    cv2.waitKey(0)	  
    cv2.destroyAllWindows() 
