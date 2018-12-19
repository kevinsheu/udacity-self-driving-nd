import numpy as np
import cv2

def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

def find_window_centroids(image, window_width, window_height, margin):
	
	window_centroids = [] # Store the (left,right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions
	
	l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
	l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
	r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
	r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
	

	window_centroids.append((l_center,r_center))
	
	for level in range(1,(int)(image.shape[0]/window_height)):
		image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)
		offset = window_width/2
		l_min_index = int(max(l_center+offset-margin,0))
		l_max_index = int(min(l_center+offset+margin,image.shape[1]))
		l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
		r_min_index = int(max(r_center+offset-margin,0))
		r_max_index = int(min(r_center+offset+margin,image.shape[1]))
		r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

		window_centroids.append((l_center,r_center))

	return window_centroids


def get_lane_pixels(warped, window_width, window_height, margin):
	window_centroids = find_window_centroids(warped, window_width, window_height, margin)

	assert(len(window_centroids) > 0)

	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)

	for level in range(0,len(window_centroids)):
		l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
		r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
		l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
		r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255


	template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
	relevantL = cv2.bitwise_and(l_points, warped)
	relevantR = cv2.bitwise_and(r_points, warped)

	zero_channel = np.zeros_like(template)
	template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
	warpage= np.dstack((warped, warped, warped))*255 
	output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) 

	# cv2.imshow('dww', output)
	# cv2.imshow('d', relevantL)
	# cv2.imshow('ef', relevantR)
	# cv2.waitKey(0)

	return relevantL, relevantR


first = True

def reset():
	global first
	first = True

def get_lanes_histogram(binary_warped, single=False):
	global first

	if first or single:
		first = False
		return get_lanes_histogram_first(binary_warped)
	else:
		return get_lanes_hist_next(binary_warped)


def get_lanes_hist_next(binary_warped):
	global left_fit
	global right_fit

	global old_leftx
	global old_lefty
	global old_rightx
	global old_righty
	
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 15
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	leftx = np.concatenate((nonzerox[left_lane_inds], old_leftx))
	lefty = np.concatenate((nonzeroy[left_lane_inds], old_lefty))
	rightx = np.concatenate((nonzerox[right_lane_inds], old_rightx))
	righty = np.concatenate((nonzeroy[right_lane_inds], old_righty))

	old_leftx = nonzerox[left_lane_inds]
	old_lefty = nonzeroy[left_lane_inds] 
	old_rightx = nonzerox[right_lane_inds]
	old_righty = nonzeroy[right_lane_inds]



	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	return leftx, lefty, rightx, righty, None


def get_lanes_histogram_first(binary_warped):
	global left_fit
	global right_fit
	global old_leftx
	global old_lefty
	global old_rightx
	global old_righty

	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	nwindows = 9
	window_height = np.int(binary_warped.shape[0]//nwindows)
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	leftx_current = leftx_base
	rightx_current = rightx_base
	margin = 50
	minpix = 50
	left_lane_inds = []
	right_lane_inds = []

	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
		(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
		(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	old_leftx = nonzerox[left_lane_inds]
	old_lefty = nonzeroy[left_lane_inds] 
	old_rightx = nonzerox[right_lane_inds]
	old_righty = nonzeroy[right_lane_inds]

	left_fit = np.polyfit(old_lefty, old_leftx, 2)
	right_fit = np.polyfit(old_righty, old_rightx, 2)


	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


	# cv2.imshow("d", out_img)
	# cv2.waitKey(0)
	# cv2.imwrite('writeup_images/histogram.jpg', out_img)



	return old_leftx, old_lefty, old_rightx, old_righty, out_img