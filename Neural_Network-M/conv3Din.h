#pragma once
#include <vector>
#include "Layer.h"

class conv3Din: public Layer
{
private:
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
public:
	conv3Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, std::vector<float**>* _matrix_in, std::vector<float**>* _out, std::vector<float**>* _error, std::vector<float**>* error_out, bool _flat, float* _flatten);
	~conv3Din();
	void feed_forward();
	void back_propagation();
	void weights_update();
	void create_errorMatrix();
};

