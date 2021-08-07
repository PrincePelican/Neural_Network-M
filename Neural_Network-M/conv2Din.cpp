#include "conv2Din.h"



conv2Din::conv2Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, float*** _matrix_in, std::vector<float**>* _out, std::vector<float**>* _error, Active_functions* _function)
{
	this->kernelSize = _kernelSize;
	this->kernelNumber = _kernelNumber;
	this->matrixSize = _matrixSize;
	this->errorSize = (matrixSize - kernelSize) + 1;
	this->matrix_in = _matrix_in;
	this->out = _out;
	this->error = _error;
	this->function = _function;

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
		matrix_operations::Conv(*matrix_in, kernels[i], (*out)[i], matrixSize, kernelSize);
	}

	function->feed_forwardM();

}

void conv2Din::back_propagation()
{
	float** conv_product = matrix_operations::createMatrix(kernelSize, kernelSize);
	//float** error_sum = matrix_operations::createMatrix(errorSize, errorSize);

	//matrix_operations::add(error_sum, (*error), errorSize);
	
	for (unsigned i{ 0 }; i < kernelNumber; ++i) {			
		matrix_operations::Conv(*matrix_in, (*error)[i], conv_product, matrixSize, errorSize);
		matrix_operations::subtract(batch[i], conv_product, kernelSize, kernelSize);
	}

	matrix_operations::clearMatrix(conv_product, kernelSize, kernelSize);
	//matrix_operations::clearMatrix(error_sum, errorSize, errorSize);


}

void conv2Din::weights_update()
{
	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::multiply(batch[i], kernelSize, kernelSize, learnRate);
		matrix_operations::add(kernels[i], batch[i], kernelSize, kernelSize);
		matrix_operations::ResetMem(batch[i], kernelSize, kernelSize);
	}
}

void conv2Din::changeLearnRate(float rate)
{
	learnRate = rate;
}

void conv2Din::initweights(Initializator::Initializators method)
{
	float initializer = 1;
	if (Initializator::Initializators::He)
	{
		initializer = Initializator::He_ini(kernelSize*kernelSize);
	}
	else if (Initializator::Initializators::Xavier)
	{
		//initializer = Initializator::Xavier_ini(kernelSize*kernelSize);
	}
	srand(static_cast <unsigned> (time(0)));
	for (unsigned x{ 0 }; x < kernelNumber; ++x) {
		for (unsigned i{ 0 }; i < kernelSize; ++i) {
			for (unsigned j{ 0 }; j < kernelSize; ++j) {
				kernels[x][i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * initializer;
			}
		}
	}
}
