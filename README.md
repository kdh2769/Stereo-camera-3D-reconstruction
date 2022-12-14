# Stereo camera depth extimation


Result
------


Introduction
------------

This project purpose to understand a stereo calibration. 
Using Stereo RGB Camera, we can estimate depth point 
 
Main idea is backprojection a point from each camera. And calculate the nearest point. 
Finally, set that point a origin point in world coordinate.

Dependencies
------------
Mono camera calibration and stereo calibration using OpenCV method. 

- Python : 3.8.2
- OpenCV : 

Manual
------
1. Set stero camera. 
2. Capture chessboard pattern more than 6 images at the same time. 
   (Remember the chess pattern size)
3. 
