#include "fully_connected.h"
#include "matrix_operations.h"
#include <iostream>

fully_connected::fully_connected(unsigned _neuronNumber, unsigned _weightsNumber, float*& _out, float*& _in, float*& _deriative, float*& _cost)
{
	this->neuronNumber = _neuronNumber;
	this->weightsNumber = _weightsNumber;
	this->out = out;
	this->in = in;
	this->deriative = _deriative;
	this->cost = _deriative;

	weights = matrix_operations::createMatrix(neuronNumber, weightsNumber);
	batch_mem = matrix_operations::createMatrix(neuronNumber, weightsNumber);
}

fully_connected::~fully_connected()
{
	for (unsigned i = 0; i < neuronNumber; ++i)
		delete weights[i];
	delete weights;
}

void fully_connected::feed_forward()
{
	matrix_operations::dot_product(out, this->weights, in, neuronNumber, weightsNumber);
}

void fully_connected::back_propagation()
{
	float* result_mul = new float[neuronNumber] {0};
	float** weights_correction = matrix_operations::createMatrix(neuronNumber, weightsNumber);
	matrix_operations::multiply(result_mul, cost, deriative, neuronNumber);
	matrix_operations::dot_product(weights_correction, result_mul, in, neuronNumber, weightsNumber);
	matrix_operations::showMatrix(weights_correction, neuronNumber, weightsNumber);
	matrix_operations::add(batch_mem, weights_correction, neuronNumber, weightsNumber);
	matrix_operations::clearMatrix(weights_correction, neuronNumber, weightsNumber);
	delete[] result_mul;
}

void fully_connected::weights_update()
{
	matrix_operations::subtract(weights, batch_mem, neuronNumber, weightsNumber);
	matrix_operations::ResetMem(batch_mem, neuronNumber, weightsNumber);
}

