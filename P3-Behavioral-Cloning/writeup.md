# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

[camera]: camera.jpg 

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 0. Introduction

In general, I followed the approach suggested in the lessons. I first implemented a network and generator, and then tried it with normal driving. After that, it became a more iterative approach - testing the car on the track, and then acquiring more data on the parts where the car did not perform well.

#### 1. An appropriate model architecture has been employed

My model is the same model from the [NVIDIA paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (model.py lines 42-62).
It consists of 3 5x5 convolution layers of stride 2, 2 3x3 convolutions of stride 1, and 3 FC layers.
The data is normalized in the model using a Keras lambda layer, and cropped appropriately so that the network can focus only on the relevant parts of the image.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (80% train, 20% test). In addition, I spent a lot of time acquiring data to ensure that the network would be generalizing instead of overfitting (more information in data acquisition strategy). Finally, I augmented my data during training in the data generator (model.py lines 65-90).

I used scikit-learn's shuffle function to randomly pick out a train and test set.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 60).

#### 4. Data acquisition strategy

Training data was chosen to keep the vehicle driving on the road. Here are the main strategies that I used:
- center lane driving (2-3 laps)
- driving backwars (1 lap)
- recovering from the left and right sides of the road, focusing on the sharper turns (many times)
- recovering on the bridge portion a couple of times

I only used images taken from the center camera. An example is shown below: 

![camera][camera]


This image is taken from one of my runs where I am trying to simulate recovering from the edge of the lane line. Ideally, if the car sees this image, it should adjust its steering angle so it will turn slightly left.


As a trained and tested the network, I would acquire more data on the parts of the track that the car was failing in. (e.g. if I am bumping into the sides of the bridge more, then I will get more data of the car driving on the bridge).

#### 5. Training Process

To train, I used a keras generator because it was impossible to load all of the training imagess into RAM. In my training generator, I also added horizontal flipping (model.py lines 65-90). This would help the network generalize more, instead of overfititng to the counterclockwise route. I used a batch size of 32, and it worked fine.

The main parts where the car kept falling off the track were the left turn next to the dirt portion of the map, and the sharp right turn that ocurred after.

#### 6. Conclusion

In conclusion, I was able to train and end-to-end self driving car, that was able to recover extrmelely well when it crossed a lane line, on both left and right sides. I used a convolutional neural network that was proven to work, and used an iterative approach to gradually acquire better data and train the car better.