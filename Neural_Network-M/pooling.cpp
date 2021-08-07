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
		pooling_forward((*matrix_in)[x], (*result_pooling)[x], matrixSize, pooling_size);
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
		Pointer.clear();
		matrix_operations::ResetMem((*error_out)[x], matrixSize, matrixSize);
		matrix_operations::pool_back((*error_in)[x], (*error_out)[x], errorSize, pooling_size);
		}
}

void pooling::weights_update()
{
}

void pooling::changeLearnRate(float rate)
{
}

void pooling::initweights(Initializator::Initializators method)
{
}


void pooling::pooling_forward(float** in, float** out, unsigned matrixSize, unsigned pooling_size)
{
	for (unsigned i{ 0 }; i < matrixSize - (pooling_size - 1); i = i + pooling_size) {	//przechodzi przez macierz i robi avg pooling
		for (unsigned j{ 0 }; j < matrixSize - (pooling_size - 1); j = j + pooling_size)
		{
			//out[i / pooling_size][j / pooling_size] = avg_feed_forward(in, i, j, pooling_size, pooling_size*pooling_size);
			out[i / pooling_size][j / pooling_size] = max_feed_forward(in, i, j, pooling_size);
		}
	}
}

void pooling::pool_back(float** in, float** out, unsigned matrixSize, unsigned pooling_size)
{
	float divider = pooling_size * pooling_size;
	unsigned outSize = matrixSize * pooling_size;
	unsigned counter = 0;
	for (unsigned i{ 0 }; i < matrixSize; ++i) {
		for (unsigned j{ 0 }; j < matrixSize; ++j) {
			//avg_back_prop(out, in, i, j, pooling_size, divider);
			max_back_prop(out, in, i, j, counter);
			counter++;
		}
	}
}

float pooling::avg_feed_forward(float** out, unsigned y, unsigned x, unsigned pooling_size, float divider)
{
	float result = 0;
	for (unsigned i{ 0 }; i < pooling_size; ++i) {
		for (unsigned j{ 0 }; j < pooling_size; ++j) {
			result += out[y + i][x + i];
		}
	}
	return result / divider;
}

void pooling::avg_back_prop(float** out, float** in, unsigned y, unsigned x, unsigned pooling_size, float divider)
{
	float result = in[y][x] / divider;
	for (unsigned i{ 0 }; i < pooling_size; ++i) {
		for (unsigned j{ 0 }; j < pooling_size; ++j) {
			out[y * 2 + i][x * 2 + j] = result;
		}
	}
}

float pooling::max_feed_forward(float** out, unsigned y, unsigned x, unsigned pooling_size)
{
	float result = 0;
	int max = out[y][x];
	Point max_point;
	for (unsigned i{ 0 }; i < pooling_size; ++i) {
		for (unsigned j{ 0 }; j < pooling_size; ++j) {
			if (out[y + i][x + i] > max) {
				max = out[y + i][x + i];
				max_point.i = y + i;
				max_point.j = x + i;
			}
		}
	}
	Pointer.push_back(max_point);
	return max;;
}



void pooling::max_back_prop(float** out, float** in, unsigned y, unsigned x, unsigned counter)
{
	float result = in[y][x];
	Point max_error = Pointer[counter];
	out[max_error.i][max_error.j] = result;
}

