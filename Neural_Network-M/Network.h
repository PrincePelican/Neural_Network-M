#pragma once
#include "Layer.h"
#include "fully_connected.h"
#include "conv2Din.h"
#include "conv3Din.h"
#include "pooling.h"
#include "matrix_operations.h"
#include "Active_functions.h"
#include <vector>


class Network
{
private:
	std::vector<Layer*> Layers;//wszystkie warstwy
	std::vector<unsigned> Sizes;//rozmiary poszczególnych warstw
	//fully connected
	std::vector<float*> result_fullyCon;
	std::vector<float*> result_funfullyCon;
	std::vector<float*> result_deriative;
	std::vector<float*> dercost_fullyCon;
	float** inData;
	float* flatten;
	//conv 3D
	std::vector<std::vector<float**>*> result_3D;
	std::vector<std::vector<float**>*> error_3D;
	void addVectors(unsigned matrixSize, unsigned vectorSize);
public:
	void add3Dconv(unsigned kernelNumber, unsigned kernelSize, bool flat = false);
	void add2Dconv(unsigned kernelNumber, unsigned kernelSize, unsigned inSize);
	void addPooling(unsigned poolingSize, bool flat = false);
	void addFullyCon(Active_functions::Active_fun function, unsigned neuronNumber, unsigned inNumber = 0);
	void changein(float** in);
	void initializatiion(Initializator::Initializators method);
	void feed_forward();
	void back_prop();

	
};

