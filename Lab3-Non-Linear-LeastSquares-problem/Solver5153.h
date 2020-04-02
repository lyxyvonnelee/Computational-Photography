#pragma once
#include "hw3_gn.h"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Solver5153 :public GaussNewtonSolver {
public:
	double solve(
		ResidualFunction *f, // Our goal is to find a group of parameter Xs, which minimizes f's value
		double *X,  // This is what the solver solves for
		GaussNewtonParams param = GaussNewtonParams(),  // this parameter tells the solver some information, such as when to terminate iteration.
		GaussNewtonReport *report = nullptr
	);
};
