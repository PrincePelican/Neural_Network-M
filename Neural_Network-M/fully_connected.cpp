#include "fully_connected.h"

fully_connected::fully_connected(unsigned _neuronNumber, unsigned _weightsNumber, unsigned _layer_n, float* _out, float* _in, float* _deriative, std::vector<float*>* _cost, bool _error3D, std::vector<float**>* _error_3D, Active_functions* _funkcje) //przypisuje potrzebne wskaŸniki tworzy macierze
{
	this->neuronNumber = _neuronNumber;
	this->weightsNumber = _weightsNumber;
	if (error3D)
	{
		this->weightsNumber = _weightsNumber * _weightsNumber * (*_error_3D).size();
	}
	this->out = _out;
	this->in = _in;
	this->deriative = _deriative;
	this->cost = _cost;
	this->layer_n = _layer_n;
	this->error3D = _error3D;
	this->error_3D = _error_3D;
	this->error3DSize = _weightsNumber;
	this->funkcje = _funkcje;

	bias = new float[neuronNumber] {0};
	batch_bias = new float[neuronNumber] {0};
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
	matrix_operations::add(out, bias, neuronNumber);

	funkcje->feed_forward();
}

void fully_connected::back_propagation() 
{
	funkcje->deriative_out();
	float* result_mul = new float[neuronNumber] {0};
	float** weights_correction = matrix_operations::createMatrix(neuronNumber, weightsNumber);
	matrix_operations::multiply(result_mul, (*cost)[layer_n], deriative, neuronNumber);	
	matrix_operations::add(batch_bias, result_mul, neuronNumber);
	matrix_operations::dot_productB((*cost)[layer_n-1], weights, result_mul, neuronNumber, weightsNumber);
	matrix_operations::dot_product(weights_correction, result_mul, in, neuronNumber, weightsNumber);
	matrix_operations::add(batch_mem, weights_correction, neuronNumber, weightsNumber);
	matrix_operations::clearMatrix(weights_correction, neuronNumber, weightsNumber);
	delete[] result_mul;
	if (error3D) {
		matrix_operations::assignTo3D((*cost)[layer_n-1], error_3D, sqrt(error3DSize/error_3D->size()));
		matrix_operations::ResetMem((*cost)[layer_n-1], neuronNumber);
	}
	matrix_operations::ResetMem(out,neuronNumber);
	matrix_operations::ResetMem((*cost)[layer_n], neuronNumber);
}

void fully_connected::weights_update()
{
	matrix_operations::multiply(batch_bias, neuronNumber, learnRate);
	matrix_operations::multiply(batch_mem, neuronNumber, weightsNumber, learnRate);
	matrix_operations::subtract(bias, batch_bias, neuronNumber);
	matrix_operations::subtract(weights, batch_mem, neuronNumber, weightsNumber);
	matrix_operations::ResetMem(batch_mem, neuronNumber, weightsNumber);
	matrix_operations::ResetMem(batch_bias, neuronNumber);
}

void fully_connected::changeLearnRate(float rate)
{
	learnRate = rate;
}

void fully_connected::initweights(Initializator::Initializators method)
{
	float initializer;
	if (Initializator::Initializators::He)
	{
		initializer = Initializator::He_ini(weightsNumber);
	}
	else if (Initializator::Initializators::Xavier)
	{
		initializer = Initializator::Xavier_ini(weightsNumber, neuronNumber);
	}
	srand(static_cast <unsigned> (time(0)));
	for (unsigned i{ 0 }; i < neuronNumber; ++i) {
		for (unsigned j{ 0 }; j < weightsNumber; ++j) {
			weights[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * initializer;
		}
	}
}


