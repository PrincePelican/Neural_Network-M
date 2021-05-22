#include <iostream>
#include "Network.h"
#include "matrix_operations.h"


int main() {



	Network A;
	A.add2Dconv(2, 3, 28);
	A.addPooling(2);
	A.add3Dconv(6, 3);
	A.addPooling(2);
	A.addFullyCon(100);
	A.addFullyCon(10);
	return 0;
}