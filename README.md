# Stereo camera depth extimation


Result
------
<p>
  <img src="./result.gif" height='60%' width ='60%'>
</p>

Introduction
------------

This project purpose to understand a stereo calibration. 
Using Stereo RGB Camera, we can estimate depth point 
 
Main idea is backprojection a point from each camera. And calculate the nearest point. 
Finally, set that point a origin point in world coordinate.

Dependencies
------------
It is tested with opencv-4.2.0 
Mono camera calibration and stereo calibration using OpenCV method. 
