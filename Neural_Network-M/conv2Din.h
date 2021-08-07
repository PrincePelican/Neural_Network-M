#pragma once
#include "Layer.h"
#include <vector>

class conv2Din: public Layer
{
private:
	float learnRate = 0.01;
	unsigned kernelSize;
	unsigned kernelNumber;
	unsigned errorSize;
	unsigned matrixSize;
	float*** matrix_in;
	std::vector<float**>* out;
	std::vector<float**>* error;
	std::vector<float**> kernels;
	std::vector<float**> batch;
	Active_functions* function;
public:
	conv2Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, float*** _matrix_in, std::vector<float**>* _out, std::vector<float**>* _error, Active_functions* _function);
	~conv2Din();
	void feed_forward();
	void back_propagation();
	void weights_update();
	void changeLearnRate(float rate);
	void initweights(Initializator::Initializators method);
};

