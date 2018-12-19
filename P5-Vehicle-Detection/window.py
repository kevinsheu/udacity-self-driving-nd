import numpy as np
import cv2

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]

	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]

	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

	window_list = []

	for ys in range(ny_windows):
		for xs in range(nx_windows):

			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]

			window_list.append(((startx, starty), (endx, endy)))

	return window_list


def get_all_windows(image, x_start_stop, y_start_stop):
	windows = []
	overlap = 0.5

	# windows += slide_window(
	# 	image, 
	# 	x_start_stop=x_start_stop, 
	# 	y_start_stop=y_start_stop, 
	# 	xy_window=(128, 128), 
	# 	xy_overlap=(overlap, overlap)
	# )

	# windows += slide_window(
	# 	image, 
	# 	x_start_stop=[None, None], 
	# 	y_start_stop=[400, 656], 
	# 	xy_window=(120, 120), 
	# 	xy_overlap=(overlap, overlap)
	# )

	# windows += slide_window(
	# 	image, 
	# 	x_start_stop=x_start_stop, 
	# 	y_start_stop=y_start_stop, 
	# 	xy_window=(112, 112), 
	# 	xy_overlap=(overlap, overlap)
	# )

	windows += slide_window(
		image, 
		x_start_stop=x_start_stop, 
		y_start_stop=y_start_stop, 
		xy_window=(104, 104), 
		xy_overlap=(overlap, overlap)
	)

	windows += slide_window(
		image, 
		x_start_stop=x_start_stop, 
		y_start_stop=y_start_stop, 
		xy_window=(96, 96), 
		xy_overlap=(overlap, overlap)
	)

	windows += slide_window(
		image, 
		x_start_stop=x_start_stop, 
		y_start_stop=y_start_stop, 
		xy_window=(80, 80), 
		xy_overlap=(overlap, overlap)
	)

	windows += slide_window(
		image, 
		x_start_stop=x_start_stop, 
		y_start_stop=y_start_stop, 
		xy_window=(72, 72), 
		xy_overlap=(overlap, overlap)
	)

	windows += slide_window(
		image, 
		x_start_stop=x_start_stop, 
		y_start_stop=y_start_stop, 
		xy_window=(64, 64), 
		xy_overlap=(overlap, overlap)
	)

	return windows
