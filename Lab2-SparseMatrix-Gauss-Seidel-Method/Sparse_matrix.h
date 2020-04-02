#pragma once
#include <vector>

class Sparsemtx {
public:

	Sparsemtx();
	double at(int row, int col);
	void insert(double val, int row, int col);
	void initializeFromVector(std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals);

private:

	std::vector<int> rowList;
	std::vector<int> dataList_col;
	std::vector<double> dataList_value;

};