#include <iostream>
#include "fully_connected.h"
#include "matrix_operations.h"

using namespace std;

int main() {
	unsigned size1 = 2;
	unsigned size2 = 3;
	unsigned size3 = 3;
	float* vector1 = new float[size1];
	float* vector2 = new float[size2];
	float* vector3 = new float[size2];
	float** out = new float*[size1];
	for (unsigned i = 0; i < size1; ++i)
		out[i] = new float[size2];
	for (unsigned i = 0; i < size1; ++i) {
		vector1[i] = i + 1;
		cout << vector1[i] << " ";
	}
	for (unsigned i = 0; i < size2; ++i) {
		vector2[i] = i + 1;
		vector3[i] = i + 1;
		cout << vector2[i] << " ";
	}
	cout << endl;
	for (unsigned i = 0; i < size1; ++i) {
		for (unsigned j = 0; j < size2; ++j) {
			out[i][j] = 0;
		}
	}
	cout << endl;


	return 0;
}