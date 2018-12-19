# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./test_images/sign0.jpg "Traffic Sign 1"
[image5]: ./test_images/sign1.jpg "Traffic Sign 2"
[image6]: ./test_images/sign2.jpg "Traffic Sign 3"
[image7]: ./test_images/sign3.jpg "Traffic Sign 4"
[image8]: ./test_images/sign4.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! My project code is located in Traffic_Sign_Classifier.ipynb (in python notebook form) and Traffic_Sign_Classifier.html (in html form). 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

In my notebook, I provide a few visualizations:
 * I show a couple of random signs.
 * I graph the distribution of the different signs in the training set.

I also parse the csv file containing the names of the different traffic signs, to get a general idea of the types of signs that are in the dataset.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

 * To preprocess the input images, the only thing I did was rescale the image values from to floats between 0 and 1.
 * I did not convert to grayscale because there was important information in the color of the images, which would have helped with classification
 * I did not use any horizontal or vertical flipping techniques, because the signs are designed to be viewed in specific orientations, and flipping signs in any way would not have helped.
 * Converted the class labels from class numbers to one-hot vectors, because this is a multi-class classification problem.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (output from keras model.summary()):
~~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 64)          36928     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 64)          36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_2 (Dense)              (None, 128)               65664     
_________________________________________________________________
dense_3 (Dense)              (None, 43)                5547      
=================================================================
Total params: 735,435
Trainable params: 735,435
Non-trainable params: 0
_________________________________________________________________
~~~~

I used 3 x [conv-conv-maxpool-dropout] blocks, and 2 fully-connected layers before the final classification layer.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train my model, I used the ADAM optimizer with defualt settings, with a batch size of 32 (the default), and with random shuffling every epoch.

I trained for only 10 epochs, and was able to get pretty good accuracy on train, val, and test set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9824
* validation set accuracy of 0.9542
* test set accuracy of 0.9495

* I used the model.fit() function from keras to report the loss and accuracy for the training and validation test sets automatically during training. I was getting good results after 10 epochs, so I stopped the training there.
* I did not try a fancy architecture (i.e. Resnet) because this classification problem was not that complex. There were only 43 classes, with good image data.
* I used a small network because the input images small (32x32) and simple.
* I used dropout to help reduce overfitting.
* I did not really do a ton of hyperparameter tuning, as the network I came up with worked pretty well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Five German traffic signs that I found on the web are shown in the ipython notebook.

 * Image 1 might be difficult to classify because of it is not a square size, so resizing it would mess up the dimensions of the sign.
 * Image 2 might be difficult to classify because of the watermark logo in the image. It also is not a square size, so resizing it would mess up the dimensions of the sign.
 * Image 3 would be straightforward to classify because it is very similar to the images in the training set.
 * Image 4 might be difficult to classify because of the watermark logos and watermark grid in the image. It also is not a square size, so resizing it would mess up the dimensions of the sign.
 * Image 5 might be difficult to classify because of the watermark logos and complete white background.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield    		  		| Yield 	 									| 
| Speed limit (60km/h)  | Priority road 								|
| Stop					| Stop											|
| Slippery Road			| Bicycles crossing     						|
| Road work	      		| Road work					 					|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is a lot lower than the test set accuracy of 94.95%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the first image, the model was extremely sure (p=0.999) that the image was a yield sign, which was correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Yield   										| 
| 3.28e-07     			| No passing for vehicles over 3.5 metric tons	|
| 1.10e-07				| Priority road									|
| 7.01e-08	      		| No vehicles					 				|
| 2.95e-08			    | Stop      									|



For the second image, the model was pretty sure (p=0.831) that the image was a priority road sign, which was wrong. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.8307         		| Priority Road   								| 
| 0.1675    			| Yield											|
| 0.0011				| End of no passing by vehicles over 3.5 metric tons|
| 0.0003      			| Roundabout mandatory			 				|
| 0.0001			    | Speed limit (50km/h)      					|


For the third image, the model was extremely sure (p=0.999) that the image was a stop sign, which was correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999         		| Stop 											| 
| 3.41e-6    			| No entry										|
| 2.25e-6				| Yield 										|
| 7.139e-7      		| Speed limit (50km/h)			 				|
| 1.85e-7			    | Speed limit (120km/h)      					|

For the fourth image, the model was not sure (p=0.536) that the image was a bicycles crossing sign, which was wrong. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.5365         		| Bicycles crossing   							| 
| 0.4559    			| Traffic signals								|
| 0.0064				| General caution								|
| 0.0007      			| Children crossing		 						|
| 0.0001			    | Road narrows on the right    					|


For the fifth image, the model was certain (p=1.0) that the image was a road work sign, which was correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work										| 
| ~0.0    				| Wild animals crossing							|
| ~0.0					| Bicycles crossing								|
| ~0.0    	  			| Beware of ice/snow		 					|
| ~0.0			  		| Road narrows on the right     				|


It is interesting to see that the wrong predictions only occured when the model was not certain (i.e. p < 0.999) about the top decision.
