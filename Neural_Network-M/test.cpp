#include <iostream>
#include "Network.h"
#include "matrix_operations.h"


int main() {
	float** in = new float*[28];
	for (unsigned i{ 0 }; i < 28; ++i) {
		in[i] = new float[28];
		for (unsigned j{ 0 }; j < 28; ++j) {
			in[i][j] = 1.0;
		}
	}


	Network A;
	A.add2Dconv(6, 3, 28);
	A.addPooling(2);
	A.add3Dconv(6, 3);
	A.addPooling(2, true);
	A.addFullyCon(Active_functions::Active_fun::RELU, 100);
	A.addFullyCon(Active_functions::Active_fun::SOFTMAX, 10);
	A.changein(in);
	A.feed_forward();
	return 0;
}