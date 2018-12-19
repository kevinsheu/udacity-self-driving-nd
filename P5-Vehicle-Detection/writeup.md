## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected. 

[//]: # "Image References"
[image1]: ./output_images/test1.jpg
[image2]: ./output_images/test2.jpg
[image3]: ./output_images/test3.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` function, which is in lines 16 through 35 of the file called `fn.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images, and exploring how they looked in varioius color spaces. Using the default RGB color spaces was defeinitely not the smart option, so I mainly explored HLS/HSV/YCrCb color spaces.

I realized early on that the color histogram of the images would not be that helpful - I also tested this out, and the inclusion of the color histogram did not help a lot with the final test accuracy of the SVM that I used.

For the parameters of the HOGs (i.e. `orientations`, `pixels_per_cell`, and `cells_per_block`), I just played around with the usual recommended values, and they turned out to work fine.

#### 2. Explain how you settled on your final choice of HOG parameters.

As I explained before, I tested out using the color histogram, and the effects of including it on the final test accuracy. Because it did not help out that much, I decided to not include it, in order to speed up the runtime of my SVM.

In the end, I decided to just use spatial binning of the YCrCb color space, and the HOG of every channel (also in the YCrCb space).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default values, and that seemed to work. I tried SVMs with other kernel functions, but the runtimes for SVMs with more complex kernel functions were very long, so running them on my machine was out of the question. 

Because the range of data from different sources (spatial binning, HOG) had different values, I had to normalize them appropriately so that the values would be uniform. I used the `StandardScaler` which is also part of the sklearn library to normalize the features.

The code for training the SVM is in lines 94-118 in file `detect.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search on the in the `slide_window` function in file `window.py`.  I used the straightforward method from the lesson, that just looped over the relevant areas and appended them to a list.

To get windows of different scales, I included a param that controlled the size of the windows. However, all were resized to 64x64.

To figure out which scales to search, I started out at 64x64 and attempted different scales. I realized that very small scales did not really help, and that cars were proportional to very large scales.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on a couple of  scales using YCrCb 3-channel HOG features plus spatially binned color, which provided a nice result.  Below are some sample images, with the green rectangles representing the detections, and the red rectangle representing the final bounding boxes (more on how the red box is calculated later).

An example with 3 vehicles:

![][image1]

An example with no vehicles:

![][image2]

An example with one vehicle:

![][image3]

One thing that I did to improve performance was hard negative mining. In the test images provided, I extracted the windows of varying scales of the portions of the images with no cars. The reasoning behind this was that I was still getting many false positives, and this hard negative mining helped alleviate this problem. The code for the hard negative mining is in lines 51-77 in the file `detect.py`.



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)

One major difference in the video implementation is that I used the Hog Sub-sampling Window Search method explained in the lesson. Doing this increased the throughput of the video pipeline, because HOGs was computed once for each scale, and the color space conversion was also happening only once.

Note: I realized that I could have lowered the `cells_per_set` variable in the `find_cars` function to 1 to improve performance, but it would have significantly increased the runtime of the video pipeline


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In addition, I summed the heatmaps over the past several frames to create a multi-frame accumulated heatmap (before calculating bounding boxes). Using this information from subsequent frames helped reduce the number of false positives even more.

You can see from the from the images above how the heatmaps help remove the redundant positives, and results in only one large bounding box for the most part.

The code for this is in the `process_image` function in `detect.py`.





---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are some of the issues that are still persisting:

- Cars on the edge of the image: When a car is on the edge of  the image, it does not always get detected fully. This may be caused by the nature of the method that I am using (i.e. using heatmaps to generate the final bounding box). To alleviate this, I could try using non-maximum suppression or other methods to draw the box around the entire car.
- Random sparse false positives: Even with the methods I used to remove false positives, there still are some occasional blips of false positive detections. This is probably due to the nature of the linear SVM that I am using. If I used a deep learning based approach using CNNs, this probably would not happen.

