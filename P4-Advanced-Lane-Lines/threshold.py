import numpy as np
import cv2

def hls_select(img, thresh=(0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255
	return binary_output


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	binary_output = np.zeros_like(scaled_sobel)

	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255


	return binary_output

def mask(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, 255)
	return cv2.bitwise_and(img, mask)