#include <iostream>
#include <opencv2/opencv.hpp>

#include "hw3_gn.h"
#include "Solver5153.h"

using namespace std;

class Ellipsoid753 : public ResidualFunction {
public:
	Ellipsoid753(std::string fName) {
		ifstream infile;
		infile.open(fName, ios::in);
		if (!infile.is_open()) {
			cout << "Data file not found!" << endl;
			exit(1);
		}
		string s;
		while (getline(infile, s)) {
			cv::Vec3d p;
			int beg = 0, i = 0;
			stringstream sstream;
			for (int k = 0; k < 3; k++) {
				for (; i < s.length() && s[i] != ' '; i++) {}
				sstream.clear();
				sstream << s.substr(beg, i - beg);
				sstream >> p[k];
				i++;
				beg = i;
			}
			datas.push_back(p);
		}
	}
	int nR() const { return datas.size(); }
	int nX() const { return 3; }
	void eval(double *R, double *J, double *X) {
		for (int i = 0; i < nR(); i++) {
			cv::Vec3d p = datas[i];
			R[i] = p[0] * p[0] / X[0] / X[0] + p[1] * p[1] / X[1] / X[1] + p[2] * p[2] / X[2] / X[2] - 1;
			for (int j = 0; j < 3; j++) {
				J[3 * i + j] = -p[j] * p[j] / X[j] / X[j] / X[j] / 2;
			}
		}
	}

private:
	vector<cv::Vec3d> datas;
};

int main() {
	Ellipsoid753 f("ellipse753.txt");
	Solver5153 solver;

	double X[3] = { 1, 1, 1 };
	GaussNewtonReport report;
	double valR = solver.solve(&f, X, GaussNewtonParams(), &report);  // varR is the value of residual vector R.

	cout << "A = " << X[0] << "\nB = " << X[1] << "\nC = " << X[2] << endl;

	string StopType[] {
		"STOP_GRAD_TOL",       // 梯度达到阈值
		"STOP_RESIDUAL_TOL",   // 余项达到阈值
		"STOP_NO_CONVERGE",    // 不收敛
		"STOP_NUMERIC_FAILURE" // 其它数值错误
	};

	cout << "Iteration time: " << report.n_iter << endl;
	cout << "Stop type: " << StopType[report.stop_type] << endl;

	getchar();

	return 0;
}
