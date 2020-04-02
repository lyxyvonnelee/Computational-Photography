
#include "Solver5153.h"

double Solver5153::solve(ResidualFunction *f, double *X, GaussNewtonParams param, GaussNewtonReport *report) {
	cv::Mat J(f->nR(), f->nX(), cv::DataType<double>::type);
	cv::Mat R(f->nR(), 1, cv::DataType<double>::type);
	cv::Mat dX(f->nX(), 1, cv::DataType<double>::type);
	param.residual_tolerance = 0.001;
	param.gradient_tolerance = 0.001;
	
	int iterCount = 0;
	double *nextX = new double[f->nX()];
	for (iterCount = 0; iterCount < param.max_iter; iterCount++) {
		f->eval((double*)R.data, (double*)J.data, X); // evaluate resudual, jacobian using X
		cv::solve(J, -R, dX, cv::DECOMP_QR); // solve the linear system, J * dX = -R
		double value = norm(R, NORM_L2);

		// Before finding the step length alpha, we need to check if the iteration has reached termination.
		if (value < param.residual_tolerance) {
			if (report) {
				report->n_iter = iterCount;
				report->stop_type = GaussNewtonReport::STOP_RESIDUAL_TOL;
			}
			return value;
		}
		if (norm(dX, NORM_L2) < param.gradient_tolerance) {
			if (report) {
				report->n_iter = iterCount;
				report->stop_type = GaussNewtonReport::STOP_GRAD_TOL;
			}
			return value;
		}

		// now, find the step length alpha, using line search algorithm.
		double alpha = 1;
		double *nextX = new double[f->nX()];
		double val_init = norm(R, NORM_L2);
		double val_old = val_init;
		double val_new;
		double if_found = false;
		while (true) {
			// calculate the nextX, by X + alpha * dX
			for (int i = 0; i < f->nX(); i++) {
				nextX[i] = X[i] + alpha * dX.at<double>(i, 0);
			}
			// evaluate
			f->eval((double*)R.data, (double*)J.data, nextX);
			val_new = norm(R, NORM_L2);

			// if alpha guarantees the descent of the value, stretch alpha.
			// else shrink it.

			if (val_new >= val_init) {
				alpha *= 0.8;
				if_found = true;  // have gone over the lowest point.
				continue;  // step length alpha should guarantee target function's descent.
			}
			else {
				if (if_found) {
					break;
				}
			}
			if (val_new < val_old) {
				alpha *= 1.6;
				val_old = val_new;
			}
			else {
				alpha *= 0.8;
				if_found = true; // Found the lowest point.
			}
		}

		// Now we've gotten the step length alpha. Add it to X, then enter the next iteration.
		for (int i = 0; i < f->nX(); i++) {
			X[i] = X[i] + alpha * dX.at<double>(i, 0);
		}
	}
	delete[] nextX;
	if (report) {
		report->n_iter = iterCount;
		report->stop_type = GaussNewtonReport::STOP_NO_CONVERGE;
	}
	return -1;
}