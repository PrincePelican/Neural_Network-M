#pragma once
#include "Layer.h"
#include "fully_connected.h"
#include "conv2Din.h"
#include "conv3Din.h"
#include "pooling.h"
#include "matrix_operations.h"
#include "Active_functions.h"
#include <chrono>
#include <iostream>
#include <fstream>


class Network
{
private:
	std::vector<Layer*> Layers;//wszystkie warstwy
	std::vector<unsigned> Sizes;//rozmiary poszczególnych warstw
	std::vector<float**>* data_Vector;
	std::vector<unsigned>* answers;
	float* target;
	unsigned answer;
	unsigned samplecount = 0;
	unsigned batchsize = 16;
	float** inData;
	//fully connected
	std::vector<float*> result_fullyCon; //wyniki po zsumowaniu
	std::vector<float*> result_funfullyCon; //wyniki po funkcji aktywacji
	std::vector<float*> result_deriative; //pochodna funkcji
	std::vector<float*> error; // b³¹d funkcji aktywacji
	float* flatten;
	//conv 3D
	std::vector<std::vector<float**>*> result_3D;
	std::vector<std::vector<float**>*> error_3D;
	void addVectors(unsigned matrixSize, unsigned vectorSize);
	void prepareTarget(unsigned size, unsigned answer);
	void updateWeights();
	void feed_forward();
	void back_prop();
	void changein(float** in);
	void ClearBuffors();
	void giveDataIn(std::vector<float**>* _dataVector, std::vector<unsigned>* _answers);
public:
	void add3Dconv(unsigned kernelNumber, unsigned kernelSize, Active_functions::Active_fun function, bool flat = false);
	void add2Dconv(unsigned kernelNumber, unsigned kernelSize, unsigned inSize, Active_functions::Active_fun function);
	void addPooling(unsigned poolingSize, bool flat = false);
	void addFullyCon(Active_functions::Active_fun function, unsigned neuronNumber, unsigned inNumber = 0);
	void changeLearnRate(float rate);
	void changeBatchSize(unsigned size);
	void initializatiion(Initializator::Initializators method);
	void Learn(std::vector<float**>* _dataVector, std::vector<unsigned>* _answers, std::ofstream& file);
	void Predict(std::vector<float**>* _dataVector, std::vector<unsigned>* _answers);

	
};

