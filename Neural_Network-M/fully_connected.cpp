#include "fully_connected.h"
#include "matrix_operations.h"

fully_connected::fully_connected(unsigned _neuronNumber, unsigned _weightsNumber, float*& _out, float* _in, float*& _deriative, float*& _cost, bool _error3D, std::vector<float**>* _error_3D, Active_functions* _funkcje) //przypisuje potrzebne wskaŸniki tworzy macierze
{
	this->neuronNumber = _neuronNumber;
	this->weightsNumber = _weightsNumber;
	if (error3D)
	{
		this->weightsNumber = _weightsNumber * _weightsNumber * (*_error_3D).size();
	}
	this->out = out;
	this->in = in;
	this->deriative = _deriative;
	this->cost = _deriative;
	this->error3D = _error3D;
	this->error_3D = _error_3D;
	this->error3DSize = _weightsNumber;
	this->funkcje = funkcje;

	weights = matrix_operations::createMatrix(neuronNumber, weightsNumber);
	batch_mem = matrix_operations::createMatrix(neuronNumber, weightsNumber);
}

fully_connected::~fully_connected()
{
	for (unsigned i = 0; i < neuronNumber; ++i)
		delete weights[i];
	delete weights;
}

void fully_connected::feed_forward()//zwraca wynik
{
	matrix_operations::dot_product(out, this->weights, in, neuronNumber, weightsNumber);
	funkcje->feed_forward();
}

void fully_connected::back_propagation()//
{
	funkcje->deriative_out();
	float* result_mul = new float[neuronNumber] {0};
	float** weights_correction = matrix_operations::createMatrix(neuronNumber, weightsNumber);
	matrix_operations::multiply(result_mul, cost, deriative, neuronNumber);
	matrix_operations::dot_product(cost, weights, result_mul, neuronNumber, weightsNumber); // tworzenie b³êdu do nastêpnej warstwy
	matrix_operations::dot_product(weights_correction, result_mul, in, neuronNumber, weightsNumber); // 
	matrix_operations::add(batch_mem, weights_correction, neuronNumber, weightsNumber);
	matrix_operations::clearMatrix(weights_correction, neuronNumber, weightsNumber);
	delete[] result_mul;
	if (error3D) {
		matrix_operations::assignTo3D(cost, error_3D, error3DSize);
	}
}

void fully_connected::weights_update()
{
	matrix_operations::multiply(batch_mem, neuronNumber, weightsNumber, 0);
	matrix_operations::subtract(weights, batch_mem, neuronNumber, weightsNumber);
	matrix_operations::ResetMem(batch_mem, neuronNumber, weightsNumber);
}


