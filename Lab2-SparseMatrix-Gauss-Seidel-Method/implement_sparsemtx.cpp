#include<algorithm>
#include<math.h>
#include "Sparse_matrix.h"
#include<iostream>

Sparsemtx::Sparsemtx() {
	
}

double Sparsemtx::at(int row, int col) {
	
	double value;	

	int irowlist=rowList[row];    //irowlist表示在rowList中下标为row的值
	int number = rowList[row + 1] - rowList[row];  //row这一行的非零元素个数为number

	if (number == 0) {
		value = 0;
		return value;
	}  //若row行没有非零元素，则返回该位置的值为零

	for (int m = 0; m < number; m++) {
		if (dataList_col[irowlist + m] == col) {
			value = dataList_value[irowlist + m];
			return value;
		}
	}
	value = 0;
	return value;
}


void Sparsemtx::insert(double val, int row, int col) {
	double insertval;
	insertval = Sparsemtx::at(row, col);
	double atvalue;
	int n;

	int jrowlist = rowList[row];    //jrowlist表示在rowList中下标为row的值
	int jnumber = rowList[row + 1] - rowList[row];  //row这一行的非零元素个数为jnumber

	if (insertval != 0) {

		if (val != 0) {
			for (n = 0; n < jnumber; n++) {
				if (dataList_col[jrowlist + n] == col) {
					dataList_value[jrowlist + n] = val;
					return;
				}
			}
		}        //当矩阵中的值以及插入值均不为零时，直接替换dataList_value中原有值

		else {
			int insert;
			for (n = 0; n < jnumber; n++) {
				if (dataList_col[jrowlist + n] == col) {
					insert = jrowlist + n;
				}
			}  
			
			int x;
			for (x = insert; x < dataList_value.size() - 1; x++) {
				dataList_value[x] = dataList_value[x + 1];  //从插入位置之后每个dataList——value向前移一格
				dataList_col[x] = dataList_col[x + 1];      //从插入位置之后每个dataList——col向前移一格
			}
			for (int y = insert + 1; y < rowList.size(); y++) {
				rowList[y] = rowList[y] - 1;		//从插入位置之后每个rowList的值减1
			}
			
			return;
		}  //当矩阵中的值不为零而插入值为零时

	}    

	else {

		if (val = 0) {
			return;
		}  //矩阵插入值与原来的值皆为零

		else {
			int input;
			for (n = 0; n < jnumber; n++) {
				if (dataList_col[jrowlist + n] == col) {
					input = jrowlist + n;
				}
			}

			int p;
			for (p = dataList_col.size(); p >input; p--) {
				dataList_value[p] = dataList_value[p - 1];  //从插入位置之后每个dataList——value向后移一格
				dataList_col[p] = dataList_col[p - 1];      //从插入位置之后每个dataList——col向后移一格
			}

			dataList_value[input] = val;					//插入位置的值变为val

			for (int q = input + 1; q < rowList.size(); q++) {
				rowList[q] = rowList[q] + 1;		//从插入位置之后每个rowList的值加1
			}

			return;
		}  //矩阵原来的值为零，插入的值不为零

	}

}


void Sparsemtx::initializeFromVector(std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals) {
	int rows_size = rows.size();
	int cols_size = cols.size();
	int vals_size = vals.size();

	if (!(rows_size == cols_size && cols_size == vals_size)) {
		std::cout << "fatal" << std::endl;
		return;
	}
	

	for (int i = 0; i < rows_size; i++) {
		int row = rows[i];
		int col = cols[i];
		double val = vals[i];
		int count = 0;
		    
			dataList_value[i] = val;
			dataList_col[i] = col;
		
			if (i = 0) {
				rowList[i] = 0;
			}
			else {

				for (int j = 0; j < rows_size; j++) {
					if (i - 1 == rows[j]) {
						count++;
						
					}                                   //第i-1行的元素个数
				}
				rowList[i] = rowList[i - 1] + count;
				/*rowList[i] = rowList[i − 1] + (number of nonzero elements on the (i − 1)th row in the original matrix)*/

			}					
	}
	int count = 0;
	for (int k = 0; k < rows_size; k++) {
		if (rows_size == rows[k]) {
			count++;
		}
	}
	rowList[rows_size] = rowList[rows_size - 1] + count;

}