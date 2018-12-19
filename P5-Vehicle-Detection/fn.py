import numpy as np
import cv2

from skimage.feature import hog

from scipy.ndimage.measurements import label

def convert_color(img, conv='RGB2YCrCb'):
	if conv == 'RGB2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	if conv == 'BGR2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	if conv == 'RGB2LUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
						vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, 
								  pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block),
								  block_norm= 'L2-Hys',
								  transform_sqrt=False, 
								  visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	# Otherwise call with one output
	else:      
		features = hog(img, orientations=orient, 
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block),
					   block_norm= 'L2-Hys',
					   transform_sqrt=False, 
					   visualise=vis, feature_vector=feature_vec)
		return features

def bin_spatial(img, size=(32, 32)):
	color1 = cv2.resize(img[:,:,0], size).ravel()
	color2 = cv2.resize(img[:,:,1], size).ravel()
	color3 = cv2.resize(img[:,:,2], size).ravel()
	return np.hstack((color1, color2, color3))
	# features = cv2.resize(img, size).ravel()
	# return features
						
def color_hist(img, nbins=32):    #bins_range=(0, 256)

	channel1_hist = np.histogram(img[:,:,0], bins=nbins)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins)

	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

	return hist_features


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size):
	
	draw_img = np.copy(img)

	
	img_tosearch = img[ystart:ystop,:,:]
	ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]


	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
	nfeat_per_block = orient*cell_per_block**2
	

	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
	

	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	boxes = []
	
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
		  
			spatial_features = bin_spatial(subimg, size=spatial_size)
			# hist_features = color_hist(subimg, nbins=hist_bins)

			test_features = X_scaler.transform(np.hstack((spatial_features, hog_features)).reshape(1, -1))    
  
			
			# test_prediction = svc.predict_prob(test_features)
			test_prediction = svc.predict(test_features)
			
			if test_prediction == 1:
			# if test_prediction[0][1] > 0.75:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				# cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),3)
				boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

	return boxes


def extract_features(image, orient, pix_per_cell, cell_per_block, spatial_size):
	image = convert_color(image, conv='BGR2YCrCb')

	ch1 = image[:,:,0]
	ch2 = image[:,:,1]
	ch3 = image[:,:,2]


	hog_feat1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
	hog_feat2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
	hog_feat3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()

	hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

	spatial_features = bin_spatial(image, size=spatial_size)
	# hist_features = color_hist(image, nbins=hist_bins)
	return np.hstack((spatial_features, hog_features))