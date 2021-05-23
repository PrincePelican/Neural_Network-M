#pragma once
#include "Layer.h"
#include <functional>
#include "matrix_operations.h"

class pooling: public Layer
{
private:
	unsigned pooling_size;
	unsigned matrixSize;
	bool flat;
	float* flatten;
	std::vector<float**>* matrix_in;
	std::vector<float**>* error_in;
	std::vector<float**>* result_pooling;
	std::vector<float**>* error_out;
public:
	pooling(unsigned _pooling_size, unsigned _matrixSize, std::vector<float**>* _matrix_in, std::vector<float**>* _error_in, std::vector<float**>* _result_pooling, std::vector<float**>* _result_error, bool flat, float* flatten);
	void feed_forward();
	void back_propagation();
	void initweights(Initializator::Initializators method);
	float avg_feed_forward(float** out, unsigned y, unsigned x, float divider);
};

