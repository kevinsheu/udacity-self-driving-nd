import tensorflow as tf
import numpy as np
import csv
import cv2
import random

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ELU, Dropout, Lambda, Cropping2D
import keras.callbacks

import sklearn.model_selection
import sklearn.utils

# load data
# the images are not loaded because limitations in RAM, and a keras generator is used instead.
def load_data(log_path_list):
	image_files = []
	angles = []
	for log_path in log_path_list:
		print "Loading {}...".format(log_path)
		csvfile = open(log_path, 'r')
		csvreader = csv.reader(csvfile)
		for line in csvreader:
			center_file = line[0]
			# left_file = line[1]
			# right_file = line[2]
			angle = float(line[3])

			image_files.append(center_file)
			angles.append(angle)

		csvfile.close()

	x = image_files
	y = np.array(angles).astype('float32')

	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
	
	return x_train, x_test, y_train, y_test


# network definition. taken from nvidia's self driving paper (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
def create_network(h, w, c):
	model = Sequential()

	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(h, w, c)))
	model.add(Lambda(lambda x: x/255. - 0.5))
	
	model.add(Conv2D(24, (5,5), padding='same', strides=(2,2), activation='relu'))
	model.add(Conv2D(36, (5,5), padding='same', strides=(2,2), activation='relu'))
	model.add(Conv2D(48, (5,5), padding='same', strides=(2,2), activation='relu'))
	model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')

	return model


# generator for training images. Loads images on the fly because of RAM limitations.
# - also introduces left/right flipping for data augmentation
def train_generator(x_train, y_train, batch_size):
	x_batch = np.zeros((batch_size, 160, 320, 3))
	y_batch = np.zeros((batch_size))
	index = 0
	while True:
		for i in range(batch_size):
			if index >= len(x_train):
				x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
				index = 0

			image = cv2.imread(x_train[index])
			x_batch[i] = image.astype('float32')
			y_batch[i] = y_train[index]

			# random horizontal flipping
			if random.randint(0,1) == 1:
				x_batch[i] = np.flip(x_batch[i], axis=1)
				y_batch[i] = -y_batch[i]


			index += 1
			

		yield x_batch.astype('float32'), y_batch.astype('float32')


# validation data generator. No augmentation.
def val_generator(x_test, y_test, batch_size):
	x_batch = np.zeros((batch_size, 160, 320, 3))
	y_batch = np.zeros((batch_size))
	index = 0
	while True:
		for i in range(batch_size):
			if index >= len(x_test):
				index = 0

			image = cv2.imread(x_test[index])
			x_batch[i] = image.astype('float32')
			y_batch[i] = y_train[index]

			index += 1
			
		yield x_batch.astype('float32'), y_batch.astype('float32')



BATCH_SIZE = 32
EPOCHS = 20

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = load_data([
		'/home/kevin/data/cloning-data-straight/driving_log.csv', 
		'/home/kevin/data/cloning-data-right-edge/driving_log.csv',
		'/home/kevin/data/cloning-data-left-edge/driving_log.csv',
		])

	print len(x_train)
	print len(x_test)
	print y_train.shape	
	print y_test.shape

	model = create_network(160, 320, 3)

	# model.fit(x=x1, y=y1, epochs=7, validation_split=0.1, shuffle=True)
	model.fit_generator(
		generator=train_generator(x_train, y_train, batch_size=BATCH_SIZE),
		steps_per_epoch=len(x_train)/BATCH_SIZE,
		epochs=EPOCHS,
		validation_data=val_generator(x_test, y_test, batch_size=BATCH_SIZE),
		validation_steps=len(x_test)/BATCH_SIZE,
	)
	model.save('model.h5')
