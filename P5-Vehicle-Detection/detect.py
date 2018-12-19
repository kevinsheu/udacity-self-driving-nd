import numpy as np
import cv2
import glob
import time
import pickle
import os
import collections

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

import window
import fn

def train():
	vehicle_filenames = glob.glob('/home/kevin/data/cars/vehicles/*/*.png')
	notvehicle_filenames = glob.glob('/home/kevin/data/cars/non-vehicles/*/*.png')

	# vehicle_filenames = vehicle_filenames[:100]
	# notvehicle_filenames = notvehicle_filenames[:100]

	all_filenames = vehicle_filenames + notvehicle_filenames
	print "{} images".format(len(all_filenames))

	X = []
	for i, filename in enumerate(all_filenames):
		image = cv2.imread(filename)

		f = fn.extract_features(
			image,
			orient=5, 
			pix_per_cell=8, 
			cell_per_block=5, 
			spatial_size=(32,32),
		)

		if i == 0:
			print "features length: {}".format(f.shape)

		X.append(f)

		if i % 1000 == 0 and i > 0:
			print "processed {} images".format(i)

	# hard negative mining

	windows = []

	neg_file = "test_images/test1.jpg"
	neg_img = cv2.imread(neg_file)
	windows += window.get_all_windows(neg_img, x_start_stop=[None,700], y_start_stop=[400,610])

	neg_file = "test_images/test2.jpg"
	neg_img = cv2.imread(neg_file)
	windows += window.get_all_windows(neg_img, x_start_stop=[None,None], y_start_stop=[400,610])

	neg_file = "test_images/test2.jpg"
	neg_img = cv2.imread(neg_file)
	windows += window.get_all_windows(neg_img, x_start_stop=[None,None], y_start_stop=[400,610])

	neg_file = "test_images/test3.jpg"
	neg_img = cv2.imread(neg_file)
	windows += window.get_all_windows(neg_img, x_start_stop=[None,750], y_start_stop=[400,610])

	neg_file = "test_images/test4.jpg"
	neg_img = cv2.imread(neg_file)
	windows += window.get_all_windows(neg_img, x_start_stop=[None,700], y_start_stop=[400,610])

	neg_file = "test_images/test5.jpg"
	neg_img = cv2.imread(neg_file)
	windows += window.get_all_windows(neg_img, x_start_stop=[None,700], y_start_stop=[400,610])

	neg_file = "test_images/test6.jpg"
	neg_img = cv2.imread(neg_file)
	windows += window.get_all_windows(neg_img, x_start_stop=[None,700], y_start_stop=[400,610])

	print "{} negative samples".format(len(windows))

	for w in windows:
		region = neg_img[w[0][1]:w[1][1], w[0][0]:w[1][0]]
		region = cv2.resize(region, (64, 64))

		f = fn.extract_features(
			region,
			orient=5, 
			pix_per_cell=8, 
			cell_per_block=5, 
			spatial_size=(32,32),
		)
		X.append(f)

	X = np.asarray(X)

	X_scaler = StandardScaler().fit(X)
	X = X_scaler.transform(X)

	Y = np.hstack((np.ones(len(vehicle_filenames)), np.zeros(len(notvehicle_filenames)+len(windows))))

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.0001)

	print X_train.shape
	print Y_train.shape
	print X_test.shape
	print Y_test.shape

	start = time.time()
	print "Training..."
	# svc = SVC(kernel='linear', probability=True)
	svc = LinearSVC()

	svc.fit(X_train, Y_train)
	print "Training done!"
	print "time taken: {}".format(time.time() - start)
	
	test_acc = svc.score(X_test, Y_test)
	print "SVM test acc: {}".format(test_acc)

	return svc, X_scaler

history = collections.deque(maxlen = 8)

def process_image(image, model, scaler):
	boxes = []
	for scale in [1, 1.15, 1.3]:
		boxes += fn.find_cars(
			img=image, 
			ystart=400, 
			ystop=610, 
			scale=scale, 
			svc=model, 
			X_scaler=scaler, 
			orient=5, 
			pix_per_cell=8, 
			cell_per_block=5, 
			spatial_size=(32,32), 
		)

	heat = np.zeros((image.shape[0], image.shape[1]))
	for d in boxes:
		heat[d[0][1]:d[1][1], d[0][0]:d[1][0]] += 1
		# cv2.rectangle(image, (d[0][0], d[0][1]), (d[1][0],d[1][1]), (0,255,0), 4)


	history.append(heat)

	total_heat = np.zeros((image.shape[0], image.shape[1]))

	for heat_t in history:
		total_heat += heat_t

	total_heat[total_heat <= int(2.5*len(history))] = 0

	labels = label(total_heat)


	for car_number in range(1, labels[1]+1):

		nonzero = (labels[0] == car_number).nonzero()

		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

		cv2.rectangle(image, box[0], box[1], (0,0,255), 2)

	return image



SVC_FILENAME = "model3.pkl"
SCALER_FILENAME = "scaler3.pkl"

if __name__ == '__main__':
	
	
	if os.path.isfile(SVC_FILENAME) and  os.path.isfile(SCALER_FILENAME):
		print "Loading trained model...",
		f = open(SVC_FILENAME, 'r')
		model = pickle.load(f)
		f.close()
		f2 = open(SCALER_FILENAME, 'r')
		scaler = pickle.load(f2)
		f2.close()
		print "done!"
	else:
		print "Training model..."
		model, scaler = train()
		f = open(SVC_FILENAME, 'w')
		pickle.dump(model, f)
		f.close()
		f2 = open(SCALER_FILENAME, 'w')
		pickle.dump(scaler, f2)
		f2.close()

	image_list = glob.glob("test_images/*.jpg")
	image_list = []
	for image_file in image_list:
		start = time.time()
		image = cv2.imread(image_file)
		res = process_image(image, model, scaler)
		
		cv2.imwrite("output_images/{}".format(image_file.split('/')[1]), res)
		print "processed {} ({}sec)".format(image_file, time.time()-start)


	video_files = ["test_video.mp4", "project_video.mp4"]

	output_files = ["test_video_output.mp4","project_video_output.mp4"]

	for ii, filename in enumerate(video_files):
		reader = cv2.VideoCapture(filename)
		fps = reader.get(cv2.CAP_PROP_FPS)
		frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

		print "total frames: {}".format(frames)

		writer = cv2.VideoWriter(
			output_files[ii], 
			cv2.VideoWriter_fourcc('m','p','4','v'), 
			fps, 
			(int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		)

		for i in range(frames):
			start = time.time()
			ret, frame = reader.read()
			if ret == True:
				output = process_image(frame, model, scaler)
				writer.write(output)
				print "processed {} frame {} of {} ({} sec)".format(filename, i+1, frames, time.time()-start)
			else:
				print "error frame {}".format(i+1)

		reader.release()
		writer.release()


