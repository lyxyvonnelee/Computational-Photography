//#include <opencv2\opencv.hpp>
//#include <cstdlib>
//#include <algorithm>
//#include <vector>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char **argv) {
//
//	if (argc != 5) {
//		cout << "invalid arguments" << endl;
//		return -1;
//	}
//
//	string input_img_name = string(argv[1]);
//	string output_img_name = string(argv[2]);
//	int w = atoi(argv[3]);
//	int h = atoi(argv[4]);
//
//	Mat kernel = Mat::ones(h * 2 + 1, w * 2 + 1, CV_32FC1);
//	kernel = kernel / ((w * 2 + 1) * (h * 2 + 1));
//
//	Mat img = imread("img.png");
//	Mat outImage;
//	filter2D(img, outImage, img.depth(), kernel);
//
//	//namedWindow("output");
//	//imshow("output", outImage);
//	//imwrite(output_img_name, outImage);
//	//waitKey(0);
//	//destroyWindow("output");
//	
//
//	return 0;
//}