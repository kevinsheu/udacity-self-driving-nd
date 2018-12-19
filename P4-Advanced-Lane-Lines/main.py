import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

import calibration
import threshold
import find_lanes


# thresholds
hls_thresh = (150, 255)
sobel_thresh = (40, 100)


# mask vertices
bottomeleft = (0+60,720)
topleft = (640-100, 470)
topright = (640+100, 470)
bottomright = (1280-60,720)
vertices = np.array([[bottomeleft, topleft, topright, bottomright]], dtype=np.int32)

# perspective transform
src = np.float32([[120, 720], [640-80, 470], [640+80, 470], [1160, 720]])
dst = np.float32([[200,720], [200,0], [1080,0], [1080,720]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# convlution params for finding lane pixels
window_width = 70 
window_height = 90 # Break image into 8 vertical layers since image height is 720
margin = 300 # How much to slide left and right for searching




print "calibrating cameras..."
ret, mtx, dist, rvecs, tvecs = calibration.calibrate_camera()


def process(img, single):
	# undistort
	check = cv2.imread('camera_cal/calibration18.jpg')
	check_und = cv2.undistort(check, mtx, dist, None, mtx)
	# cv2.imwrite('writeup_images/checkerboard_undistort.jpg', check_und)

	# cv2.imshow('orig', img)
	img = cv2.undistort(img, mtx, dist, None, mtx)
	# cv2.imwrite('writeup_images/image_undistort.jpg', img)
	# cv2.imshow('undis', img)
	# get threshold
	hls = threshold.hls_select(img, thresh=hls_thresh)
	sobelx = threshold.abs_sobel_thresh(img, orient='x', thresh=sobel_thresh)
	filtered = cv2.bitwise_or(hls, sobelx)
	# cv2.imwrite('writeup_images/threshold.jpg', filtered)

	# cv2.imshow('d1', filtered)
	# cv2.waitKey(0)

	# Mask relevant areas
	masked = threshold.mask(filtered, vertices)
	# cv2.imwrite('writeup_images/masked.jpg', masked)

	# cv2.imshow('dwef', masked)
	# cv2.waitKey(0)

	# warp persepctive
	warped = cv2.warpPerspective(masked, M, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)
	# cv2.imshow('dwefwef', masked)
	# cv2.imwrite('writeup_images/warped.jpg', warped)

	# get pixels for left/right lane lines

	left_x, left_y, right_x, right_y, hist_img = find_lanes.get_lanes_histogram(warped, single)

	left_fit = np.polyfit(left_y, left_x, 2)
	right_fit = np.polyfit(right_y, right_x, 2)

	left_fit_y = np.linspace(100, 720, num=720-100, endpoint=False, dtype=int)
	left_fit_x = left_fit[0]*left_fit_y**2 + left_fit[1]*left_fit_y + left_fit[2]
	left_fit_x = left_fit_x.astype(int)

	right_fit_y = np.linspace(100, 720, num=720-100, endpoint=False, dtype=int)
	right_fit_x = right_fit[0]*right_fit_y**2 + right_fit[1]*right_fit_y + right_fit[2]
	right_fit_x = right_fit_x.astype(int)

	y_eval = 719

	lane_center = ( (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]) + (right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]) ) / 2.
	car_offset = lane_center - img.shape[1]/2
	car_offset_m = car_offset * 3.7 / 830.



	# turn radius calculation
	ym_per_pix = 30./720. # meters per pixel in y dimension
	xm_per_pix = 3.7/700. # meters per pixel in x dimension
	y_eval = 720*ym_per_pix-1
	left_fit_m = np.polyfit(left_y*ym_per_pix, left_x*xm_per_pix, 2)
	right_fit_m = np.polyfit(right_y*ym_per_pix, right_x*xm_per_pix, 2)
	left_curverad = ((1 + (2*left_fit_m[0]*y_eval + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
	right_curverad = ((1 + (2*right_fit_m[0]*y_eval + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])
	avg_curverad = (left_curverad + right_curverad) / 2.
	avg_curverad_m = avg_curverad * 3.7 / 128.

	# print left_fit_x
	# print right_fit_x

	# print "Turn radius: {} m".format(avg_curverad_m)
	# print "car offset: {} m".format(car_offset_m)

	left_pixels_fit = np.zeros_like(img)
	left_pixels_fit[left_fit_y, left_fit_x, 0] = 255

	right_pixels_fit = np.zeros_like(img)
	right_pixels_fit[right_fit_y, right_fit_x, 0] = 255

	lane_pixels_inv = cv2.warpPerspective(left_pixels_fit+right_pixels_fit, Minv, (right_pixels_fit.shape[1], right_pixels_fit.shape[0]), flags=cv2.INTER_LINEAR)
	kernel = np.ones((9,9), np.uint8)
	lane_pixels_inv = cv2.dilate(lane_pixels_inv, kernel, iterations = 1)
	# cv2.imwrite('output_images/lane_pixels.jpg', lane_pixels_inv)


	drawn = cv2.add(img, lane_pixels_inv)
	# cv2.imwrite('output_images/lane_pixels_plotted.jpg', drawn)


	# cv2.imshow('pixels', left_pixels_fit+right_pixels_fit)
	# plt.imshow(hist_img)
	# plt.plot(left_fit_x, left_fit_y, color='yellow')
	# plt.plot(right_fit_x, right_fit_y, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.savefig('writeup_images/line_fit.jpg')


	y_all = np.concatenate((left_fit_y, np.flip(right_fit_y, 0)))
	x_all = np.concatenate((left_fit_x, np.flip(right_fit_x, 0)))

	points = np.stack((x_all, y_all)).transpose()

	lane_shaded = np.zeros_like(img)
	cv2.fillConvexPoly(lane_shaded, points, (0,255,0))
	lane_shaded_inv = cv2.warpPerspective(lane_shaded, Minv, (lane_shaded.shape[1], lane_shaded.shape[0]), flags=cv2.INTER_LINEAR)

	mask = lane_shaded_inv[:,:,1].astype(float)
	mask[mask > 0] = 0.5
	mask = np.expand_dims(mask,axis=2)

	output = (1-mask) * img + mask * lane_shaded_inv
	output = output.astype(np.uint8)

	cv2.putText(output, 'Turn radius: {:.2f}m'.format(avg_curverad_m), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	if car_offset_m > 0:
		side = 'right'
	else:
		side = 'left'
	cv2.putText(output, 'Car offset: {:.2f}m to the {}'.format(car_offset_m, side), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

	return output




test_images = glob.glob('test_images/s*1.jpg')
for test_file in test_images:
	print "Processing {}...".format(test_file)
	img = cv2.imread(test_file)
	cv2.imshow("img", img)
	output = process(img, single=True)
	cv2.imshow('w', output)
	# cv2.imwrite('output_images/' + test_file.split('/')[1], output)
	if cv2.waitKey(0) == ord('q'):
		break

# cv2.destroyAllWindows()
# quit()



# video_names = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
# output_names = ['project_video_output.mp4', 'challenge_video_output.mp4', 'harder_challenge_video_output.mp4']
video_names = ['project_video.mp4']
output_names = ['project_video_output.mp4']

video_names = []

for ii, filename in enumerate(video_names):
	reader = cv2.VideoCapture(filename)
	fps = reader.get(cv2.CAP_PROP_FPS)
	frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

	print "total frames: {}".format(frames)

	writer = cv2.VideoWriter(
		output_names[ii], 
		cv2.VideoWriter_fourcc('M','J','P','G'), 
		fps, 
		(int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	)

	find_lanes.reset()
	for i in range(frames):
		print "processing {} frame {} of {}".format(filename, i, frames)
		ret, frame = reader.read()
		if ret:
			output = process(frame, single=False)
			writer.write(output)

	reader.release()
	writer.release()
	break