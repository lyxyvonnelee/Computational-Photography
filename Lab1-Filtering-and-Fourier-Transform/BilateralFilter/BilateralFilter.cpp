#include<opencv2\opencv.hpp>
#include<vector>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#define pi 3.1415926


using namespace cv;
using namespace std;

int main(int argc, char **argv) {
	
	string input_img_name = "img.png";
	string output_img_name = "BilateralFilter.png";
	Mat img = imread("img.png");
	

	img.convertTo(img, CV_32FC3);
	cvtColor(img, img, CV_BGR2GRAY);
	Mat oimg(img.size(), CV_32F);
	
	//IplImage *img = cvLoadImage("img.jpg", 1);
	//IplImage *img_8U = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	//IplImage *img_64F = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, 1);
	//cvCvtColor(img, img_8U, CV_BGR2GRAY); 
	//cvConvertScale(img_8U, img_64F);   // //8U转64F
	//cv::Mat(img);


	//double sigma_s=1.0;
	//double sigma_r=1.0;
	//int w = floor(5 * sigma_s);
	//int h = floor(5 * sigma_s);
	//
	//double csum = 0;
	//double wsum = 0;
	//double gauss_s;
	//double gauss_r;
	//double ws;

	float sigma_s = 1.0;
	float sigma_r = 0.5*256;
	int w = floor(5 * sigma_s);
	int h = floor(5 * sigma_s);
	
	float csum = 0;
	float wsum = 0;
	float gauss_s;
	float gauss_r;
	float ws;


	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0 ; j <img.cols; ++j) {
			csum = 0;
			wsum = 0;
			for (int u = -h; u <= h; ++u) {
				for (int v = -w; v <= w; ++v) {

					int x = i+u, y = j+v;
					if (x < 0) {
						x = 0;
					}
					else if (x >= img.rows) {
						x = img.rows - 1;
					}
					if (y < 0) {
						y = 0;
					}
					else if (y >= img.cols) {
						y = img.cols - 1;
					}

					/*double r = abs(img.at<double>(i, j) - img.at<double>( x, y));*/
					float r = abs(img.at<float>(i, j) - img.at<float>(x, y));
					
					gauss_s = 1.0 / (2 * pi*sigma_s*sigma_s)*exp(-(u*u + v*v) / (2 * sigma_s*sigma_s));//1.0/(2*pi*sigma_s*sigma_s)*
					gauss_r = 1.0 / (2 * pi*sigma_s*sigma_s)*exp(-(r*r) / (2 * sigma_r*sigma_r));//1.0 / (2 * pi*sigma_s*sigma_s)*
					ws = abs(gauss_s*gauss_r);
					/*csum += ws*img.at<double>( x,  y);*/
					
					csum += ws*img.at<float>( x, y);
					wsum += ws;
					
				}
			}

			///*img.at<double>(i, j) = csum / wsum;*/

			oimg.at<float>(i, j) = csum / wsum;
			
			/*if (img.at<float>(i,j) != 0) {
				img.at<float>(i, j) = 0;
			}*/
			/*cout << img.at<float>(i, j) << endl;*/

		}
	}

	oimg.convertTo(oimg, CV_8UC1);
	img.convertTo(img, CV_8UC1);
	cvtColor(oimg, oimg, CV_GRAY2BGR);
	cvtColor(img, img, CV_GRAY2BGR);
	imwrite(output_img_name, oimg);
	
			imshow("out", oimg);
			imshow("put", img);
			waitKey(0);
			

	return 0;
}