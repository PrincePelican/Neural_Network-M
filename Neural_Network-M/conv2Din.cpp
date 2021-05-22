#include "conv2Din.h"
#include "matrix_operations.h"


conv2Din::conv2Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, float** _matrix_in, std::vector<float**>* _out, std::vector<float**>* _error)
{
	this->kernelSize = _kernelSize;
	this->kernelNumber = _kernelNumber;
	this->matrixSize = _matrixSize;
	this->errorSize = matrixSize - (kernelSize + 1);
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

conv2Din::~conv2Din()
{
	for (float**& in : kernels) {
		matrix_operations::clearMatrix(in, kernelSize, kernelSize);
	}
}


void conv2Din::feed_forward()
{
	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::Conv(matrix_in, kernels[i], (*out)[i], matrixSize, kernelSize);
	}
}

void conv2Din::back_propagation()
{
	float** conv_product = matrix_operations::createMatrix(kernelSize, kernelSize);
	float** error_sum = matrix_operations::createMatrix(errorSize, errorSize);

	matrix_operations::add(error_sum, (*error), errorSize);
	
	for (unsigned i{ 0 }; i < kernelNumber; ++i) {			//przyjrzeæ siê ten sam in za ka¿dym razem
		matrix_operations::Conv(matrix_in, error_sum, conv_product, matrixSize, errorSize);
		matrix_operations::subtract(batch[i], conv_product, kernelSize, kernelSize);
	}

	matrix_operations::clearMatrix(conv_product, kernelSize, kernelSize);
	matrix_operations::clearMatrix(error_sum, errorSize, errorSize);


}

void conv2Din::weights_update()
{
	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::multiply(batch[i], kernelSize, kernelSize, 0);
		matrix_operations::add(kernels[i], batch[i], kernelSize, kernelSize);
		matrix_operations::ResetMem(batch[i], kernelSize, kernelSize);
	}
}
