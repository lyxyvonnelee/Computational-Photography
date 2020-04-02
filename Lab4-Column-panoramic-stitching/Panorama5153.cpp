#include "Panorama5153.h"
#include "hw6_pa.h"
#include "opencv2\xfeatures2d.hpp"
#include <string.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


bool Panorama::makePanorama(std::vector<cv::Mat>& img_vec, cv::Mat& img_out, double f)
{
	
	//projection from 3D to 2D
	double  r=f;
	double x, y, x1, y1;
	cv::Mat X,X0;
	int img_num = 12; //size of vector img_vec
	std::vector<cv::Mat> destImg_vec; //store 12 destImg

	for (int i = 0; i < img_num; ++i) {
		X = Mat(img_vec[i]);
		cv::Mat destImg(X.size(), X.type());


		int xmax = X.cols;
		int ymax = X.rows;

		for (int m = 0; m < xmax; ++m) {
			for (int n = 0; n < ymax; ++n) {						

				x1 = f*tan((m-xmax/2) / r);
				y1 = (n-ymax/2) / r*sqrt(x1*x1 + f*f); //3D

				x1 += xmax / 2;
				y1 += ymax / 2;
				
				//make sure the point is inside the original image
				if (!((int)x1 > 0 && (int)x1 < X.cols - 2 && (int)y1>0 && (int)y1 < X.rows - 2)) {
					continue;
				}
				
				//using bilinear interpolation
				double dx = x1 - (int)x1;
				double dy = y1 - (int)y1;

				double weight_tl = (1.0 - dx)*(1.0 - dy);
				double weight_tr = dx*(1.0 - dy);
				double weight_bl = (1.0 - dx)*dy;
				double weight_br = dx*dy;				

				uchar rvalue = weight_tl*X.at<Vec3b>((int)y1, (int)x1)[0]+weight_tr*X.at<Vec3b>((int)y1,(int)(x1+1))[0]
					           +weight_bl*X.at<Vec3b>((int)(y1+1),(int)x1)[0]+weight_br*X.at<Vec3b>((int)(y1+1),(int)(x1+1))[0];

				uchar gvalue= weight_tl*X.at<Vec3b>((int)y1, (int)x1)[1] + weight_tr*X.at<Vec3b>((int)y1, (int)(x1 + 1))[1]
					+ weight_bl*X.at<Vec3b>((int)(y1 + 1), (int)x1)[1] + weight_br*X.at<Vec3b>((int)(y1 + 1), (int)(x1 + 1))[1];

				uchar bvalue= weight_tl*X.at<Vec3b>((int)y1, (int)x1)[2] + weight_tr*X.at<Vec3b>((int)y1, (int)(x1 + 1))[2]
					+ weight_bl*X.at<Vec3b>((int)(y1 + 1), (int)x1)[2] + weight_br*X.at<Vec3b>((int)(y1 + 1), (int)(x1 + 1))[2];

				
				destImg.at<Vec3b>(n, m)[0] = rvalue;
				destImg.at<Vec3b>(n, m)[1] = gvalue;
				destImg.at<Vec3b>(n, m)[2] = bvalue;
				
			}
		}

		/*cv::namedWindow("OUTPUT");
		cv::imshow("OUTPUT", destImg);
		cv::waitKey(0);*/

		destImg_vec.push_back(destImg);
		
	}

	

	//detect the keypoints using SURF Detector
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	Mat img_dest1,img_dest2;

	img_dest1 = Mat(destImg_vec[0]);
	img_dest2 = Mat(destImg_vec[1]);
	

	for (int j = 0; j < img_num-1; ++j) {
		img_dest2 = Mat(destImg_vec[j+1]);
		vector<KeyPoint> keypoint1, keypoint2;
		
		
		detector->detect(img_dest1, keypoint1);
		detector->detect(img_dest2, keypoint2);

		
		//compute the descriptors
		Mat descriptor1, descriptor2;
		detector->compute(img_dest1, keypoint1, descriptor1);
		detector->compute(img_dest2, keypoint2, descriptor2);
		

		//use BruteForce to match 
		BFMatcher matcher;
		vector<DMatch> match1;

		matcher.match(descriptor1, descriptor2, match1);
		sort(match1.begin(), match1.end()); //sort out matching points

		vector<DMatch> well_match;
		for (int k = 0; k < 1; ++k) {
			well_match.push_back(match1[k]);
			//cout << match1[k].distance << endl;
		}

		

		vector<Point2f> imagePoints1, imagePoints2;
		for (int i = 0; i<10; i++)
		{
			imagePoints1.push_back(keypoint1[match1[i].queryIdx].pt);
			imagePoints2.push_back(keypoint2[match1[i].trainIdx].pt);
		}


		////draw matches
		//Mat imgmatch;
		//drawMatches(img_dest1, keypoint1, img_dest2, keypoint2, well_match, imgmatch);

		////show the result
		//namedWindow("result");
		//imshow("result", imgmatch);
		//waitKey(0);

		//calculate average distance
		double distance,adistance=0;
		int count = 0;
		for (int m = 0; m <match1.size() ; ++m) {
			distance = match1[m].distance;
			adistance += distance;
			count++;
		}

		adistance /= count;
		
		/*cout << adistance << endl;
		getchar();*/
		
		double dx=0, dy=0,each_dx,each_dy;

		//calculate average dx,dy
		for (int n = 0; n < 10; ++n) {
			each_dx = imagePoints2[n].x - imagePoints1[n].x;
			each_dy = imagePoints2[n].y - imagePoints1[n].y;
			dx += each_dx;
			dy += each_dy;
		}


		dx = abs((int)dx/10);
		dy = abs((int)dy/10);

		/*cout << dx << endl;
		cout << dy << endl;
		getchar();*/

		Point2i a;
		a.x = (int)dx;
		a.y = (int)dy;
		
				
		Mat stitch = linearStitch(img_dest1, img_dest2, a);
		imshow("拼接结果", stitch);
		waitKey(0);
		

		/*Mat combine;
		hconcat(img_dest1, img_dest2, combine);

		imshow("out", combine);
		waitKey(0);*/

		img_dest1 = Mat(stitch);
		



	}






	return true;
}



/*渐入渐出拼接
*参数列表中，img1,img2为待拼接的两幅图像，a为偏移量
*返回值为拼接后的图像
*/
Mat linearStitch(Mat img, Mat img1, Point2i a)
{
	int d, ms, ns;
	d = img.cols - a.x;//过渡区宽度
	ms = img.rows - abs(a.y);//拼接图行数
	ns = img.cols + a.x;//拼接图列数

	//if (a.x > 0) {
	//	 d = img.cols - a.x;//过渡区宽度
	//	ms = img.rows - abs(a.y);//拼接图行数
	//	ns = img.cols + a.x;//拼接图列数
	//}
	//else{
	//	d = img.cols + a.x;//过渡区宽度
	//	ms = img.rows - abs(a.y);//拼接图行数
	//	ns = img.cols - a.x;//拼接图列数
	//}
	
	Mat stitch = Mat::zeros(ms, ns, CV_8UC3);
	//拼接
	Mat_<Vec3b> ims(stitch);
	Mat_<Vec3b> im(img);
	Mat_<Vec3b> im1(img1);

	if (a.y >= 0)
	{
		Mat roi1(stitch, Rect(0, 0, a.x, ms));
		img(Range(a.y, img.rows), Range(0, a.x)).copyTo(roi1);
		Mat roi2(stitch, Rect(img.cols, 0, a.x, ms));
		img1(Range(0, ms), Range(d, img1.cols)).copyTo(roi2);
		for (int i = 0; i < ms; i++)
			for (int j = a.x; j < img.cols; j++) {
				

				ims(i, j)[0] = uchar((img.cols - j) / float(d)*im(i + a.y, j)[0] + (j - a.x) / float(d)*im1(i, j - a.x)[0]);
				ims(i, j)[1] = uchar((img.cols - j) / float(d)*im(i + a.y, j)[1] + (j - a.x) / float(d)*im1(i, j - a.x)[1]);
				ims(i, j)[2] = uchar((img.cols - j) / float(d)*im(i + a.y, j)[2] + (j - a.x) / float(d)*im1(i, j - a.x)[2]);

			}

	}
	else
	{
		Mat roi1(stitch, Rect(0, 0, a.x, ms));
		img(Range(0, ms), Range(0, a.x)).copyTo(roi1);
		Mat roi2(stitch, Rect(img.cols, 0, a.x, ms));
		img1(Range(-a.y, img.rows), Range(d, img1.cols)).copyTo(roi2);
		for (int i = 0; i < ms; i++)
			for (int j = a.x; j < img.cols; j++) {
			
				
				ims(i, j)[0] = uchar((img.cols - j) / float(d)*im(i, j)[0] + (j - a.x) / float(d)*im1(i + abs(a.y), j - a.x)[0]);
				ims(i, j)[1] = uchar((img.cols - j) / float(d)*im(i, j)[1] + (j - a.x) / float(d)*im1(i + abs(a.y), j - a.x)[1]);
				ims(i, j)[2] = uchar((img.cols - j) / float(d)*im(i, j)[2] + (j - a.x) / float(d)*im1(i + abs(a.y), j - a.x)[2]);
			}
	}


	return stitch;
}