//#include<opencv2\opencv.hpp>
//#include<vector>
//#include<cstdlib>
//#include<algorithm>
//#include<math.h>
//#define pi 3.1415926
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char **argv) {
//	string input_img_name = "bridge-compare.png";
//	string output_img_name = "bridge-compare.png";
//	double sigma = 10;
//
//	//Mat img = imread("bridge-compare.png");
//	Mat img = imread("img.png");
//
//	int w = floor(5 * sigma);
//	
//	Mat kernel(2*w+1, 2*w+1, CV_64FC1);
//	double sum = 0;
//	for (int i = -w; i <= w; ++i) {
//		for (int j = -w; j <= w; ++j) {
//			double point = 1.0 / (2 * pi*sigma*sigma)*exp(-(i*i + j*j) / (2 * sigma*sigma));
//			kernel.at<double>(i+w, j+w) = point;
//			sum += point;
//		}
//	}
//	kernel /= sum;
//	imshow("input", img);
//	Mat out;
//	filter2D(img, out, img.depth(), kernel);
//
//	imwrite(output_img_name, out);
//	namedWindow("output");
//	imshow("output", out);
//	waitKey(0);
//	destroyWindow("output");
//
//	return 0;
// }