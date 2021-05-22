#include "pooling.h"

pooling::pooling(unsigned _pooling_size, unsigned _matrixSize, std::vector<float**>* _matrix_in, std::vector<float**>* _error_in, std::vector<float**>* _result_pooling, std::vector<float**>* _result_error, bool _flat, float* _flatten)
{
	this->pooling_size = _pooling_size;
	this->matrixSize = _matrixSize;
	this->matrix_in = _matrix_in;
	this->error_in = _error_in;
	this->result_pooling = _result_pooling;
	this->error_out = _result_error;
	this->flat = _flat;
	this->flatten = _flatten;
}

void pooling::feed_forward()
{
	for (unsigned x{ 0 }; x < matrix_in->size(); ++x) {
		matrix_operations::pooling((*matrix_in)[x], (*result_pooling)[x], matrixSize, pooling_size);
	}
	if (flat) {
		matrix_operations::assingToflatten(result_pooling, flatten, matrixSize / pooling_size);
	}
}

void pooling::back_propagation()
{
	float divider = pooling_size * pooling_size;
	unsigned errorSize = matrixSize / pooling_size;
	for (unsigned x{ 0 }; x < matrix_in->size(); ++x) {
		matrix_operations::pool_back((*error_in)[x], (*error_in)[x], matrixSize / pooling_size, pooling_size);
		}
	}


float pooling::avg_feed_forward(float** out, unsigned y, unsigned x, float divider)
{
	float result = 0;
	for (unsigned i{ 0 }; i < pooling_size; ++i) {
		for (unsigned j{ 0 }; j < pooling_size; ++j) {
			result += out[y + i][x + i];
		}
	}
	return result / divider;
}

