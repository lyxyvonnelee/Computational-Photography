#pragma once
#include <opencv2/opencv.hpp>
#include "hw6_pa.h"
#include <stdlib.h>
#include <iostream>


using namespace std;
using namespace cv;

class Panorama :public CylindricalPanorama {
public:
	virtual bool makePanorama(
		std::vector<cv::Mat>& img_vec, cv::Mat& img_out, double f
	) ;
	
};

Mat linearStitch(Mat img, Mat img1, Point2i a);