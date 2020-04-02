#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <opencv2\bgsegm.hpp>
#include "hw6_pa.h"
#include "Panorama5153.h"
//#pragma comment (lib,"opencv_bgsegm320.lib")


using namespace std;
using namespace cv;

int main() {
	std::string prefix = "./img/dataset1/DSC01"; //filename of photos
	int img_num = 12; //number of photos
	int begNo = 538;

	std::vector<cv::Mat> img_vect; 
	cv::Mat img_out;


	//read photos to img_vec
	for (int i = 0; i < img_num; i++) {
		std::string path = prefix + std::to_string(begNo + i) + ".JPG";
		/*std::cout << path << std::endl;*/
		cv::Mat img = cv::imread(path);
		img_vect.push_back(img);
	}

	//read f
	double f=512.89;


	/*for (int i = 0; i < img_vect.size(); i++) {
		cv::imshow("test", img_vect[i]);
		cv::waitKey(0);
	}
	getchar(); */

	Panorama makepano;
	bool sign = makepano.makePanorama( img_vect, img_out, f);


	return 0;
}