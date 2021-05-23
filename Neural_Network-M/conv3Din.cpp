#include "conv3Din.h"
#include "matrix_operations.h"

conv3Din::conv3Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, std::vector<float**>* _matrix_in, std::vector<float**>* _out, std::vector<float**>* _error, std::vector<float**>* error_out, bool _flat, float* _flatten)
{
	this->kernelSize = _kernelSize;
	this->kernelNumber = _kernelNumber;
	this->matrixSize = _matrixSize;
	this->errorSize = matrixSize - (kernelSize + 1);
	this->matrix_in = _matrix_in;
	this->out = _out;
	this->error = _error;
	this->flat = _flat;
	this->flatten = _flatten;

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
	matrix_operations::add(sum_in, (*matrix_in), matrixSize); // dodaje macierz wszystkie pola macierzy 2D

	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::Conv(sum_in, kernels[i], (*out)[i], matrixSize, kernelSize);
	}

	matrix_operations::clearMatrix(sum_in, matrixSize, matrixSize);
	if (flat) {
		matrix_operations::assingToflatten(out, flatten, matrixSize - (kernelSize - 1));
	}
}

void conv3Din::back_propagation()
{
	float** conv_product = matrix_operations::createMatrix(kernelSize, kernelSize);
	float** matrix_in_sum = matrix_operations::createMatrix(matrixSize, matrixSize);

	matrix_operations::add(matrix_in_sum, (*matrix_in), matrixSize);

	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::Conv(matrix_in_sum, (*error)[i], conv_product, matrixSize, errorSize);
		matrix_operations::subtract(batch[i], conv_product, kernelSize, kernelSize);
	}

	create_errorMatrix();
	matrix_operations::clearMatrix(conv_product, kernelSize, kernelSize);
	matrix_operations::clearMatrix(matrix_in_sum, matrixSize, matrixSize);
}

void conv3Din::weights_update()
{
	for (unsigned i{ 0 }; i < kernelNumber; ++i) {
		matrix_operations::multiply(batch[i], kernelSize, kernelSize, 0);
		matrix_operations::add(kernels[i], batch[i], kernelSize, kernelSize);
		matrix_operations::ResetMem(batch[i], kernelSize, kernelSize);
	}
}

void conv3Din::create_errorMatrix()
{
	for (unsigned x{ 0 }; x < kernels.size(); ++x) {
		unsigned error_s = errorSize + (kernelSize - 1);  // rozmiar nowej macierzy

		matrix_operations::rotate180(kernels[x], kernelSize);

		int startx = 0 - (kernelSize - 1);
		int starty = 0 - (kernelSize - 1);
		int endy = errorSize;
		int endx = errorSize;
		int zero = 0;
		for (int i = startx; i < endx; ++i) {
			for (int j = starty; j < endy; ++j) {
				int wp = 0 - std::min(j, zero);
				int kp = 0 - std::min(i, zero);
				int mp1 = j + kernelSize - errorSize; // rozmiar filtra - rozmiar b³êdu 
				int mp2 = i + kernelSize - errorSize;
				int limitw = kernelSize - std::max(mp1, zero);
				int limitk = kernelSize - std::max(mp2, zero);
				for (int w = wp; w < limitw; ++w) {
					for (int k = kp; k < limitk; ++k) {
						(*error_out)[x][j - starty][i - startx] += kernels[x][w][k] * (*error)[x][w+j][k+i];
					}
				}
			}
		}
		matrix_operations::rotate180(kernels[x], kernelSize);
	}
}

void conv3Din::initweights(Initializator::Initializators method)
{
	float initializer = 1;
	if (Initializator::Initializators::He)
	{
		initializer = Initializator::He_ini(kernelSize * kernelSize);
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

