# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

I followed the guideline from the lesson when designing my pipeline. My pipeline consisted of 6 main steps:

  * **1. Convert image to grayscale, so each pixel is a single channel value.**

  * **2. Apply gaussian blurring to suppress noise and spurious gradients that exist in the image.**

  This results in less erronious edges detected by the OpenCV edges detection. I used kernel sizes given in the lesson, and these worked fine. 

  * **3. Get edges using OpenCV Canny edge detection algorithm.**

  I used thresholds from the lesson, and these worked fine.

  * **4. Mask out irrelevant edges which would be ouside of the expected lane line area.**

  I started out with values from the lesson, and gradually shrunk the masked area to make my pipeline more accurate.

  * **5. Determine lines from using OpenCV Hough transform.**

  I used parameters from the lesson, and these worked fine.


In order to draw a single line on the left and right lanes, I modified the draw_lines() function:

  * First, I categorized every detected line segment as part of the right lane line, or part of the left lane line. I did this by calculating the slope of the line, and assigning it as a left/right line depending on the sign of the slope.

  * Then, I used every point in each set of left/right lines to calculate the average line.

  * Finally, I extrapolated each line to the bottom of the image.

### 2. Identify potential shortcomings with your current pipeline

  * **Hard coded mask areas**

  If this pipeline was used on a car where the camera was placed in a different location, it might have incorrect results, because the points of the masked polygon are hard coded in the pipeline. 

  * **Curved lines**

  My pipeline would not perform well on very curvy roads. This is because I use functions from OpenCV that are designed to detect straight lines. 

  * **Latency**
  The code is not very optimized, and would not work fast enough to be useful in a real-time driving scenario.


### 3. Suggest possible improvements to your pipeline

 * To make the pipeline more portable to different camera setups, I should use calculate the position using ratios. This would also allow the pipeline to be used on images/videos of varying aspect ratios.

 * To adapt the pipeline for use in scenarios with curved lane lines, I could use more generalzied curve detection algorithms and/or DL methods to detect the lane lines.

 * To make the entire pipeline run faster, I could use vectorized numpy functions, or rewrite it in C/C++.

 