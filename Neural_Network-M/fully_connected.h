#pragma once
#include "Layer.h"


class fully_connected: public Layer
{
private:
	float** weights;
	float** batch_mem;
	float* out;
	float* in;
	float* deriative;
	float* cost;
	float bias;
	unsigned neuronNumber;
	unsigned weightsNumber;
public:
	fully_connected(unsigned _neuronNumber, unsigned _weightsNumber, float*& _out, float*& _in, float*& _deriative, float*& _cost);
	~fully_connected();
	void feed_forward();
	void back_propagation();
	void weights_update();
};

