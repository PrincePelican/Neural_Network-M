#pragma once
#include "Layer.h"
#include "Active_functions.h"
#include <vector>


class fully_connected: public Layer
{
private:
	float learnRate = 0.01;
	float** weights;
	float** batch_mem;
	float* out;
	float* in;
	float* deriative;
	std::vector<float*>* cost;
	float* bias;
	float* batch_bias;
	bool error3D;
	unsigned layer_n;
	unsigned error3DSize;
	unsigned neuronNumber;
	unsigned weightsNumber;
	std::vector<float**>* error_3D;
	Active_functions* funkcje;
public:
	fully_connected(unsigned _neuronNumber, unsigned _weightsNumber, unsigned _layer_n, float* _out, float* _in, float* _deriative, std::vector<float*>* _cost, bool _error3D, std::vector<float**>* _error_3D, Active_functions* _funkcje);
	~fully_connected();
	void feed_forward();
	void back_propagation();
	void weights_update();
	void changeLearnRate(float rate);
	void initweights(Initializator::Initializators method);
};

