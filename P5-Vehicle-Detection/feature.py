import numpy as np
import cv2

import skimage.feature


def convert(img, color_space='YCrCb'):
	feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	return feature_image

def bin_spatial(img, color_space='RGB', size=(32, 32)):
	# Convert image to new color space (if specified)
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else:
		feature_image = np.copy(img)             
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(feature_image, size).ravel() 
	# Return the feature vector
	return features

def get_hog(image):
	features = skimage.feature.hog(
		image,
		orientations=5,
		pixels_per_cell=(8, 8),
		cells_per_block=(5, 5),
		transform_sqrt=True,
		feature_vector=True
	)

	return features


def extract_features(image, features):

	# features = bin_spatial(image, color_space='YCrCb', size=(32, 32))

	for c in range(image.shape[2]):
		features = np.concatenate((features, get_hog(image[:,:,c])))

	return features