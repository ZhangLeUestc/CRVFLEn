
This software is an imaplentation of the tracking method described in Visual Tracking with Convolutional Random Vector
Functional Link Network.

The code is based on the following work:

MEEM: Robust Tracking via Multiple Experts using Entropy Minimization", Jianming Zhang, Shugao Ma, Stan Sclaroff, ECCV, 2014.

The code is maintained by Zhang Le. If you have questions, please contact zhang.le@adsc.com.sg/lzhang027@e.ntu.edu.sg


June. 2016


This code has been tested on 64-bit Windows with OpenCV 2.40+ on CPU. We have not tested it on GPU but it is easy to extend based on it.


Installation:

0. You should have OpenCV 2.40+ and MatConvNet (we use 1.0-beta7)  installed.
1. Unzip the files to <install_dir>.
2. Launch Matlab.
3. Go to <install_dir>\mex, and open "compile.m".
4. Change the OpenCV inlude and lib directory to yours, ans save.
5. run "compile" in Matlab.
4. Go back to <install_dir>, and run "demo".


How to use:

Just insert it into the visual tracking benchmark: 

Wu, Yi, Jongwoo Lim, and Ming-Hsuan Yang. "Online object tracking: A benchmark." Proceedings of the IEEE conference on computer vision and pattern recognition. 2013.


