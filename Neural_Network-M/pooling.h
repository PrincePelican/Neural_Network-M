#pragma once
#include "Layer.h"
#include <functional>
#include "matrix_operations.h"

class pooling
{
private:
	unsigned pooling_size;
	std::function<float(float**, unsigned, unsigned, unsigned)> pooling_fun;
public:
	pooling(unsigned _pooling_size, std::function<float(float**, unsigned, unsigned, unsigned)> f);
	void feed_forward(float**& matrix_in, float**& out, unsigned sizeY, unsigned sizeX);
	void back_propagation();
};

