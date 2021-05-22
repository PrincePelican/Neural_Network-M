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
	std::vector<Layer*> Layers;//wszystkie warstwy
	std::vector<unsigned> Sizes;//rozmiary poszczególnych warstw
	std::vector<std::function<float*(float*)>> active_functions;	//vektor przechowujacy funkcje aktywacji
	//fully connected
	std::vector<float*> result_fullyCon;
	std::vector<float*> result_funfullyCon;
	std::vector<float*> result_deriative;
	std::vector<float*> dercost_fullyCon;
	float* flatten;
	//conv 3D
	std::vector<std::vector<float**>> result_3D;
	std::vector<std::vector<float**>> error_3D;
public:
	void add3Dconv(unsigned kernelNumber, unsigned kernelSize, bool flat = false);
	void add2Dconv(unsigned kernelNumber, unsigned kernelSize, unsigned inSize);
	void addPooling(unsigned poolingSize, bool flat = false);
	void addFullyCon(unsigned neuronNumber, unsigned inNumber = 0);
	void addVectors(unsigned matrixSize, unsigned vectorSize);

	
};

