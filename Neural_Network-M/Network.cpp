#include "Network.h"
#include <iostream>
#include <chrono>

void Network::add3Dconv(unsigned kernelNumber, unsigned kernelSize, bool flat) //dodaje konwolucje wejœcia 3D
{
	unsigned matrixSize = Sizes.back(); //pobiera rozmiar wejœcia z poprzedniej warstwy
	unsigned outSize = matrixSize - (kernelSize - 1);	//oblicza rozmiar wyjœcia
	unsigned SizetoPush = outSize;
	if (flat)//w przypadku kiedy nastêpny wymiar sieci oczekuje rozmiaru 1D sp³aszcza 3D
	{
		flatten = new float[kernelNumber * outSize * outSize]{0};
		result_funfullyCon.push_back(flatten);
		SizetoPush = kernelNumber * outSize * outSize;
	}
	addVectors(outSize, kernelNumber, result_3D);
	addVectorsError(matrixSize, kernelNumber, error_3D[error_3D.size()-1]);
	addVectorsError(matrixSize, kernelNumber, error_3D[error_3D.size() - 2]);
	error_3D.push_back(new std::vector<float**>);
	Layers.push_back(new conv3Din(kernelSize, kernelNumber, matrixSize, result_3D[result_3D.size()-2], result_3D.back(), error_3D.back(), error_3D[error_3D.size() - 2], flat, flatten));
	Sizes.push_back(SizetoPush);
}

void Network::add2Dconv(unsigned kernelNumber, unsigned kernelSize, unsigned inSize) //dodaje konwolucje wejœcia 2D
{
	unsigned outSize = inSize - (kernelSize - 1); // oblicza rozmiar wyjœæia konwolucji
	addVectors(outSize, kernelNumber, result_3D);
	error_3D.push_back(new std::vector<float**>);
	Layers.push_back(new conv2Din(kernelSize, kernelNumber, inSize, &inData, result_3D[0], error_3D[0]));
	Sizes.push_back(outSize);
}

void Network::addPooling(unsigned poolingSize, bool flat)
{
	unsigned inSize = Sizes.back();
	unsigned outSize = inSize / poolingSize;
	unsigned SizetoPush = outSize;
	if (flat)//w przypadku kiedy nastêpny wymiar sieci oczekuje rozmiaru 1D sp³aszcza 3D
	{
		flatten = new float[outSize * outSize * result_3D.back()->size()]{0};
		result_funfullyCon.push_back(flatten);
		SizetoPush = outSize * outSize * result_3D.back()->size();
	}
	addVectors(outSize, result_3D.back()->size(), result_3D);
	error_3D.push_back(new std::vector<float**>);
	Layers.push_back(new pooling(poolingSize, inSize, result_3D[result_3D.size() - 2], error_3D.back(), result_3D.back(), error_3D[error_3D.size() - 2], flat, flatten));
	Sizes.push_back(SizetoPush);
}

void Network::addFullyCon(Active_functions::Active_fun function, unsigned neuronNumber, unsigned inNumber)
{
	bool error3D = false;
	unsigned SizeWeights = inNumber;
	if (inNumber == 0) {
		SizeWeights = Sizes.back();
		if (!error_3D.empty() && dercost_fullyCon.empty()) {
			error3D = true;
			unsigned kernels_number = error_3D[error_3D.size() - 3]->size();
			addVectorsError(sqrt(SizeWeights/kernels_number), kernels_number, error_3D[error_3D.size()-1]);
			addVectorsError(sqrt(SizeWeights / kernels_number), kernels_number, error_3D[error_3D.size() - 2]);
		}
	}
	if (dercost_fullyCon.empty()) {
		dercost_fullyCon.push_back(new float[SizeWeights] {0});
	}
	result_fullyCon.push_back(new float[neuronNumber] {0});//tworzy potrzebne macierze do przekazania wskaŸników
	result_funfullyCon.push_back(new float[neuronNumber] {0});
	result_deriative.push_back(new float[neuronNumber] {0});
	dercost_fullyCon.push_back(new float[neuronNumber] {0});
	Active_functions* functions = new Active_functions(neuronNumber, result_fullyCon.back(), result_funfullyCon.back(), result_deriative.back(), function, &correct);


	Layers.push_back(new fully_connected(neuronNumber, SizeWeights, dercost_fullyCon.size()-1, result_fullyCon.back(), result_funfullyCon[result_funfullyCon.size() - 2], result_deriative.back(), &dercost_fullyCon, error3D, error_3D.back(), functions));
	Sizes.push_back(neuronNumber);
}

void Network::changein(float** in)
{
	inData = in;
}

void Network::ClearBuffors()
{
	unsigned x = Layers.size() - result_fullyCon.size();
	for (unsigned i = 0; i < result_fullyCon.size(); ++i) {
		matrix_operations::ResetMem(result_fullyCon[i], Sizes[x]);
		matrix_operations::ResetMem(result_funfullyCon[i+1], Sizes[x]);
		++x;
	}
}

void Network::changeLearnRate(float rate)
{
	for (unsigned i{ 0 }; i < Layers.size(); ++i)
	{
		Layers[i]->changeLearnRate(rate);
	}
}

void Network::initializatiion(Initializator::Initializators method)
{
	for (unsigned i{ 0 }; i < Layers.size(); ++i)
	{
		Layers[i]->initweights(method);
	}
}

void Network::feed_forward()
{
	for (unsigned i{ 0 }; i < Layers.size(); ++i)
		Layers[i]->feed_forward();
}

void Network::back_prop()
{
	for (unsigned i{ Layers.size() - 1 }; i > 0; --i)
		Layers[i]->back_propagation();
}

void Network::updateWeights() {
	for (unsigned i{ 0 }; i < Layers.size(); ++i)
		Layers[i]->weights_update();
}

void Network::giveDataIn(std::vector<float**>* _dataVector, std::vector<unsigned>* _answers)
{
	this->data_Vector = _dataVector;
	this->answers = _answers;
}

void Network::Learn(std::vector<float**>* _dataVector, std::vector<unsigned>* _answers)
{
	giveDataIn(_dataVector, _answers);
	std::ios::sync_with_stdio(false);
	target = new float[Sizes.back()];
	float counter = 0;
	float good = 0;
	auto start = std::chrono::high_resolution_clock::now();
	
	for (unsigned i{ 0 }; i < data_Vector->size(); ++i) {
		samplecount++;
		changein((*data_Vector)[i]);
		feed_forward();
		correct = (*answers)[i];
		//matrix_operations::showVector(result_funfullyCon.back(), 10);
		prepareTarget(Sizes.back(), (*answers)[i]);
		matrix_operations::subtract(dercost_fullyCon.back(), result_funfullyCon.back(), target, Sizes.back());
		unsigned Networ_pred = matrix_operations::chooseMax(result_funfullyCon.back(), Sizes.back());
		//std::cout << "Answer:" << (*answers)[i] << " " << "Network_prediction:" << Networ_pred << std::endl;
		//std::cout << "[" << counter << "/" << (*data_Vector).size() << "] Acc:" << (good / counter * 100.0f) << "%" << std::endl;
		if ((*answers)[i] == Networ_pred) ++good;
		back_prop();
		if (samplecount == batchsize) {
			updateWeights();
			samplecount = 0;
		}
		++counter;
	}
	std::cout << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "[" << counter << "/" << (*data_Vector).size() << "] Acc:" << (good / counter * 100.0f) << "%" << std::endl;
	std::cout << "Epoch Time:" << elapsed.count() << "sec" << std::endl;
	delete[] target;
}

void Network::Predict(std::vector<float**>* _dataVector, std::vector<unsigned>* _answers)
{
	giveDataIn(_dataVector, _answers);
	target = new float[Sizes.back()];
	float counter = 0;
	float good = 0;
	auto start = std::chrono::high_resolution_clock::now();
	for (unsigned i{ 0 }; i < data_Vector->size(); ++i) {
		ClearBuffors();
		changein((*data_Vector)[i]);
		feed_forward();
		unsigned Networ_pred = matrix_operations::chooseMax(result_funfullyCon.back(), Sizes.back());
		if ((*answers)[i] == Networ_pred) ++good;
		++counter;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << std::endl;
	std::cout << "[" << counter << "/" << (*data_Vector).size() << "]" << std:: endl;
	std::cout << "Test_Acc:" << good / counter * 100.0f << "% Time:" << elapsed.count() << "sec" << std::endl;
	delete[] target;
}


void Network::addVectors(unsigned matrixSize, unsigned vectorSize, std::vector<std::vector<float**>*> &vector)
{
	std::vector<float**>* new_vec = new std::vector<float**>;
	for (unsigned i{ 0 }; i < vectorSize; ++i) { // tworzy potrzebne macierze
		new_vec->push_back(matrix_operations::createMatrix(matrixSize, matrixSize));
	}
	vector.push_back(new_vec);
}

void Network::addVectorsError(unsigned matrixSize, unsigned vectorSize, std::vector<float**>* vector) {
	if (vector->empty()) {
		for (unsigned i{ 0 }; i < vectorSize; ++i) { // tworzy potrzebne macierze
			vector->push_back(matrix_operations::createMatrix(matrixSize, matrixSize));
		}
	}
}

void Network::prepareTarget(unsigned size, unsigned answer)
{
	for (unsigned i{ 0 }; i < size; ++i) {
		target[i] = 0.0f;
		if (i == answer) target[i] = 1.0f;
	}
}	





