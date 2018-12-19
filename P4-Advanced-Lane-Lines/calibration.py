import numpy as np
import glob
import cv2

def calibrate_camera():

	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


	objpoints = []
	imgpoints = []

	images = glob.glob('camera_cal/calibration*.jpg')


	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

			# img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
			# cv2.imshow('img',img)
			# cv2.waitKey(500)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	return ret, mtx, dist, rvecs, tvecs