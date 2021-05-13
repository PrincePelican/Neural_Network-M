#pragma once
#include "Layer.h"
#include "fully_connected.h"
#include "conv2Din.h"
#include "conv3Din.h"
#include "pooling.h"
#include "matrix_operations.h"
#include <vector>

class Network
{
private:
	std::vector<Layer*> Layers;
	//fully connected
	std::vector<float*> result_fullyCon;
	std::vector<float*> result_funfullyCon;
	std::vector<float*> result_deriative;
	std::vector<float*> dercost_fullyCon;
	//conv 2D
	std::vector<float**> result_conv2D;
	std::vector<float**> error_conv2D;
	//conv 3D
	std::vector<std::vector<float**>> result_conv3D;
	std::vector<std::vector<float**>> error_conv3D;
	//pooling
	std::vector<std::vector<float**>> result_pooling;
public:
	void add3Dconv(unsigned kernelNumber, unsigned kernelSize);
	void add2Dconv(unsigned kernelNumber, unsigned kernelSize);
	void addPooling(unsigned poolingSize);
	void addFullyCon(unsigned neuronNumber, unsigned inNumber);

	
};

