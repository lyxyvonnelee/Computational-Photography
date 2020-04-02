#include<opencv2\opencv.hpp>
#include<vector>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include "Sparse_matrix.h"
#include<iostream>

using namespace cv;
using namespace std;


int main(int argc, char* argv[]) {

	vector<int> rows;
	vector<int> cols;
	vector<double> vals;
	vector<double> b;
	int scanf = 0,cscanf=0,vscanf=0,bscanf=0;
	/*int count = 0;*/
	int n = 0;

	while (cin >> scanf) {
		n++;
		rows.push_back(scanf);
	}
	while (cin >> cscanf) {
		cols.push_back(cscanf);
	}
	while (cin >> vscanf) {
		vals.push_back(vscanf);
	}
	while (cin >> bscanf) {
		b.push_back(bscanf);
	}
	//while (cin >> scanf) {
	//	count++;
	//	if (count % 4  == 1)
	//	{
	//		rows.push_back(scanf);
	//		n++;
	//	}
	//	else if (count % 4 == 2)
	//	{
	//		cols.push_back(scanf);
	//	}
	//	else if (count % 4 == 3)
	//	{
	//		vals.push_back(scanf);
	//	}
	//	else {
	//		b.push_back(scanf);
	//	}
	//	if (count == 4) count = 0;
	//}

	vector<double> output;

	for (int i = 0; i < b.size(); i++) {
		output[i] = 1.0;
	}

	Sparsemtx A;
	A.initializeFromVector(rows, cols, vals);

	while (1) {
		for (int j = 0; j < n; j++) {

			double sum = 0;

			for (int k = 0; k < n; k++) {
				if (j != k) {
					sum = sum + output[j] * A.at(j, k);
				}
			}

			output[j] = 1 / A.at(j, j)*(b[j] - sum);
		}

		vector<double> R1 ;
		R1[0] = 0;
		for (int p = 0; p < b.size(); p++) {
			for (int q = 0; q < b.size(); q++) {
				R1[p] += A.at(p, q)*output[q];
			}
			R1[p] = abs(R1[p] - b[p]);
		}

		int signal;
		for (int x = 0; x < b.size(); x++) {
			if (R1[x]<0.001) {
				signal=1;
			}
		}
		if (signal = 1) {
			break;
		}

	}

	for (int m = 0; m < output.size(); m++) {
		cout << output[m] << " " << endl;
	}
	
	return 0;
}