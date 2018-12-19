## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"
[image0]: ./output_images/checkerboard_orig.jpg
[image1]: ./output_images/checkerboard_undistort.jpg
[image2]: ./output_images/image.jpg
[image3]: ./output_images/image_undistort.jpg
[image4]: ./output_images/threshold.jpg
[image5]: ./output_images/masked.jpg
[image6]: ./output_images/warped.jpg
[image7]: ./output_images/histogram.jpg
[image8]: ./output_images/line_fit.jpg
[image9]: ./output_images/straight_lines1.jpg
[image10]: ./output_images/lane_pixels.jpg
[image11]: ./output_images/lane_pixels_plotted.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

## 1. Provide a Writeup / README that includes all the rubric points annes # through # ofd how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 



Original Image:

![][image0]

Undistorted:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The camera calibration calculations were only performed once, at the beginning of the pipeline. The distortion matrix was used on to distort every image, again using the `cv2.undistort` function. Here is an example of undistortion on a test image:

Original:

![alt text][image2]

Undistorted:

![][image3]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in file `threshold.py`). I thresholded the S channel between 150 and 255, and thresholded the Sobel gradient in the `x` direction only (because the lane lanes are vertical). Here's an example of my output for this step.

![alt text][image4]

Then, I masked out the region of the image that was relevant for lane line finding. The `mask` function is also located in `threshold.py`. I used hard-coded values for the vertices of the trapezoid region. Below is an example of the output after the mask:

![][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I calculated the transformation matrix `M` once at the beginning of the pipeline (lines 24-27 in `main.py`). In line 68, I use the `cv2.warpPerspective` function to perform the perspective transform, again using values that were hard-coded for simplicity. Below is an example of the warped image:

![][image6]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the histogram method that was mentioned in the lesson to obtain the pixels that would be considered part of the lane line, and thus be used in the polynomial line fit. Before using this method, I tried the convolution method using sliding windows, but it was not giving good results. The code for this is in `get_lanes_histogram()` function in `find_lanes.py`. 

In addition, my video pipeline used the positions of the lane from the previous frame to augment the calculation of the lane lines for the next lane, as described in the lesson. I use a flag in my `get_lanes_histogram` function as well as global variables to determine whether or not to use the method to find the lane lines from scratch, or using information from the previous frame.

The function for finding lanes from scratch is `get_lanes_histogram_first()`, and the function for finding lane lanes using information from previous frames is in `get_lanes_hist_next()`.

This image below shows the parts of the image that were identified as the lane line pixels, as well as the polynomial lines that were fitted for the left and right lanes.

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the method explained in the lesson to calculate the turn radius. I scaled the x and y dimensions from pixels to meters, and re-calculated the polynomial line fit. Then, I used the formula to calculate the radius of curvature in meters. The code is located in lines 96-104 in `main.py`.

I used a different method to calculate the position of the vehicle with respect to the center. I first calculated the average pixel width of the lane, and scaled the x position of the car so that the width would be 3.7m. Assuming the camera was in the middle of the car, I was able to calculate the horizontal offset of the car. The code is located at lines 87-91 in `main.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 112-125 in `main.py` .  Here is an image showing the plotted lane pixels using the polynomial fitted function transformed back into the original perspective (I used `cv2.dilate` to make the pixels easier to see):

![][image10]



And here is an image of the lane pixels plotted on the input image:

![][image11]





And here is an example of my final result on a test image (code for generating final result is in lines 137-160 in `main.py`):

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One issue that I was able to overcome was the problem of finding the relevant lane pixels, when the binary thresholded image had a significant amount of noise. I tried a couple of different approaches, and then realized that the histogram method mentioned in the lesson was most straightforward. I also improved it a little by reducing the margin of the search window, in an attempt to make the search more active.

One issue that still exists is that the calculation of the turn radius is not accurate. This is because the entire turn radius is extrapolated from an extremely small portion of the road, and minute offsets would considerably alter the final radius calculation. To improve this, I could use a bigger portion of the road, and try to extract all of the lane pixels (even the ones far away). This would give the regression more data, and the final radius calculation would be more accurate.

