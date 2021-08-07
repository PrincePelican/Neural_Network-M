#pragma once
#include <vector>
#include "Layer.h"

class conv3Din: public Layer
{
private:
	float learnRate = 0.01;
	unsigned kernelSize;
	unsigned kernelNumber;
	unsigned errorSize;
	unsigned matrixSize;
	bool flat;
	float* flatten;
	std::vector<float**>* matrix_in;
	std::vector<float**>* out;
	std::vector<float**>* error;
	std::vector<float**>* error_out;
	std::vector<float**> kernels;
	std::vector<float**> batch;
	Active_functions* function;
public:
	conv3Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, std::vector<float**>* _matrix_in, std::vector<float**>* _out, std::vector<float**>* _error, std::vector<float**>* _error_out, bool _flat, float* _flatten, Active_functions* _function);
	~conv3Din();
	void feed_forward();
	void back_propagation();
	void weights_update();
	void changeLearnRate(float rate);
	void initweights(Initializator::Initializators method);
	void create_errorMatrix(std::vector<float**>* error_in);
};

