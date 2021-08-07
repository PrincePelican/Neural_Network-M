#pragma once
#include "Layer.h"
#include <functional>
#include "matrix_operations.h"

class pooling: public Layer
{
	struct Point {
		unsigned i, j;
	};
private:
	unsigned pooling_size;
	unsigned matrixSize;
	bool flat;
	float* flatten;
	std::vector<Point> Pointer;
	std::vector<float**>* matrix_in;
	std::vector<float**>* error_in;
	std::vector<float**>* result_pooling;
	std::vector<float**>* error_out;
public:
	pooling(unsigned _pooling_size, unsigned _matrixSize, std::vector<float**>* _matrix_in, std::vector<float**>* _error_in, std::vector<float**>* _result_pooling, std::vector<float**>* _result_error, bool flat, float* flatten);
	void feed_forward();
	void back_propagation();
	void weights_update();
	void changeLearnRate(float rate);
	void initweights(Initializator::Initializators method);
	void pooling_forward(float** in, float** out, unsigned matrixSize, unsigned pooling_size);
	float max_feed_forward(float** out, unsigned y, unsigned x, unsigned pooling_size);
	float avg_feed_forward(float** out, unsigned y, unsigned x, unsigned pooling_size, float divider);
	void pool_back(float** in, float** out, unsigned matrixSize, unsigned pooling_size);
	void avg_back_prop(float** out, float** in, unsigned y, unsigned x, unsigned pooling_size, float divider);
	void max_back_prop(float** out, float** in, unsigned y, unsigned x, unsigned counter);
};

