#pragma once
#include "Layer.h"
#include <vector>

class conv2Din: public Layer
{
private:
	unsigned kernelSize;
	unsigned kernelNumber;
	unsigned errorSize;
	unsigned matrixSize;
	float** matrix_in;
	std::vector<float**>* out;
	std::vector<float**>* error;
	std::vector<float**> kernels;
	std::vector<float**> batch;
public:
	conv2Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, float** _matrix_in, std::vector<float**>* _out, std::vector<float**>* _error);
	~conv2Din();
	void feed_forward();
	void back_propagation();
	void weights_update();
};

