//#include<opencv2\opencv.hpp>
//#include<vector>
//#include<cstdlib>
//#include<algorithm>
//#include<math.h>
//#include <cstdlib>  // C standard library
//
//using namespace cv;
//using namespace std;
//
//int asd(const void* a, const void* b) {
//	return *(uchar*)a - *(uchar*)b;
//}
//
//int main(int argc,char **argv) {
//	string input_img_name = "img.png";
//	string output_img_name = "MedianFilter.png";
//	int w = 3;
//	int h = 3;
//		
//	Mat img = imread("img.png");
//
//	cvtColor(img, img, CV_BGR2GRAY);
//
//	uchar* pixels = new uchar[(2 * w + 1)*(2 * h + 1)];
//	for (int i = 0; i <img.rows ; ++i) {
//		for (int j = 0; j < img.cols; ++j) {
//			// malloc(C)  new delete(C++)
//			int cnt = 0;  // count pixels 
//			for (int u = i - h; u <= i + h; u++) {
//				for (int v = j - w; v <= j + w; v++) {
//					int y = u, x = v;
//					if (x < 0) {
//						x = 0;
//					}
//					else if (x >= img.cols) {
//						x = img.cols-1;
//					}
//					if (y < 0) {
//						y = 0;
//					}
//					else if (y >= img.rows) {
//						y = img.rows-1;
//					}
//					pixels[cnt] = img.at<uchar>(y, x);
//					cnt++;
//				}
//			}
//			// sort pixel
//			qsort(pixels, cnt, sizeof(uchar), asd);
//			img.at<uchar>(i, j) = pixels[cnt / 2];
//		}
//	}
//	delete[] pixels;
//
//	imshow("out", img);
//	waitKey(0);
//
//	cvtColor(img, img, CV_GRAY2BGR);
//	imwrite(output_img_name, img);
//
//
//	return 0;
//}