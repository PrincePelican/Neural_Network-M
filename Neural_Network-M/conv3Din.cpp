#include "conv3Din.h"
#include "matrix_operations.h"

conv3Din::conv3Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, unsigned _errorSize, std::vector<float**>& _matrix_in, std::vector<float**>& _out, std::vector<float**>& _error)
{
	this->kernelSize = _kernelSize;
	this->kernelNumber = _kernelNumber;
	this->matrixSize = _matrixSize;
	this->errorSize = _errorSize;
	this->matrix_in = _matrix_in;
	this->out = _out;
	this->error = _error;

	kernels.resize(kernelNumber);
	batch.resize(kernelNumber);

	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		kernels[i] = matrix_operations::createMatrix(kernelSize, kernelSize);
		batch[i] = matrix_operations::createMatrix(kernelSize, kernelSize);
	}
}

conv3Din::~conv3Din()
{
	for (float**& in : kernels) {
		matrix_operations::clearMatrix(in, kernelSize, kernelSize);
	}
}


void conv3Din::feed_forward()
{
	float** sum_in = matrix_operations::createMatrix(matrixSize, matrixSize);
	matrix_operations::add(sum_in, matrix_in, matrixSize); // dodaje macierz wszystkie pola macierzy 2D

	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::Conv(sum_in, kernels[i], out[i], matrixSize, kernelSize);
	}

	matrix_operations::clearMatrix(sum_in, matrixSize, matrixSize);
}

void conv3Din::back_propagation()
{
	float** conv_product = matrix_operations::createMatrix(kernelSize, kernelSize);
	float** matrix_in_sum = matrix_operations::createMatrix(matrixSize, matrixSize);

	matrix_operations::add(matrix_in_sum, matrix_in, matrixSize);

	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::Conv(matrix_in_sum, error[i], conv_product, matrixSize, errorSize);
		matrix_operations::subtract(batch[i], conv_product, kernelSize, kernelSize);
	}
}