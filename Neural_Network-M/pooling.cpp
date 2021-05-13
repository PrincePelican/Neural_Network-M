#include "pooling.h"

pooling::pooling(unsigned _pooling_size, std::function<float(float**, unsigned, unsigned, unsigned)> f)
{
	this->pooling_size = _pooling_size;
	this->pooling_fun = f;
}

void pooling::feed_forward(float**& matrix_in, float**& out, unsigned sizeY, unsigned sizeX)
{
	for (unsigned i{0}; i < sizeY - (pooling_size - 1); i = i + pooling_size) {	//przechodzi przez macierz i robi avg pooling na 4 wartoœciach
		for (unsigned j{0}; j < sizeX - (pooling_size - 1); j = j + pooling_size)
		{
			out[i/pooling_size][j/pooling_size] = pooling_fun(matrix_in, i, j, pooling_size);
		}
	}
}

void pooling::back_propagation()
{
}
