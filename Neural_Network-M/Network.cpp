#include "Network.h"

void Network::add3Dconv(unsigned kernelNumber, unsigned kernelSize, bool flat) //dodaje konwolucje wejœcia 3D
{
	unsigned matrixSize = Sizes.back(); //pobiera rozmiar wejœcia z poprzedniej warstwy
	unsigned outSize = matrixSize - (kernelSize + 1);	//oblicza rozmiar wyjœcia
	if (flat)//w przypadku kiedy nastêpny wymiar sieci oczekuje rozmiaru 1D sp³aszcza 3D
	{
		flatten = new float[kernelNumber * outSize * outSize];
	}
	addVectors(outSize, kernelNumber);
	Layers.push_back(new conv3Din(kernelSize, kernelNumber, matrixSize, &result_3D[result_3D.size()-2], &result_3D.back(), &error_3D.back(), &error_3D[error_3D.size() - 2], flat, flatten));
	Sizes.push_back(outSize);
}

void Network::add2Dconv(unsigned kernelNumber, unsigned kernelSize, unsigned inSize) //dodaje konwolucje wejœcia 2D
{
	unsigned outSize = inSize - (kernelSize + 1); // oblicza rozmiar wyjœæia konwolucji
	addVectors(outSize, kernelNumber);
	Layers.push_back(new conv2Din(kernelSize, kernelNumber, inSize, nullptr, &result_3D.back(), &error_3D.back()));
	Sizes.push_back(outSize);
}

void Network::addPooling(unsigned poolingSize, bool flat)
{
	unsigned inSize = Sizes.back();
	unsigned outSize = inSize / poolingSize;
	addVectors(outSize, result_3D.back().size());
	Layers.push_back(new pooling(poolingSize, inSize, &result_3D[result_3D.size() - 2], &error_3D.back(), &result_3D.back(), &error_3D[error_3D.size() - 2], flat, flatten));
	Sizes.push_back(outSize);
}

void Network::addFullyCon(unsigned neuronNumber, unsigned inNumber)
{
	result_fullyCon.push_back(new float[neuronNumber]{0});//tworzy potrzebne macierze do przekazania wskaŸników
	result_funfullyCon.push_back(new float[neuronNumber] {0});
	result_deriative.push_back(new float[neuronNumber]{0});
	dercost_fullyCon.push_back(new float[neuronNumber]{0});
	bool error3D = false;
	if (inNumber == 0) {
		inNumber = Sizes.back();
		if (!error_3D.empty()) {
			error3D = true;
		}
	}
	Layers.push_back(new fully_connected(neuronNumber, inNumber, result_fullyCon.back(), flatten, result_deriative.back(), dercost_fullyCon.back(), error3D, &error_3D.back()));
	Sizes.push_back(neuronNumber);
}

void Network::addVectors(unsigned matrixSize, unsigned vectorSize)
{
	std::vector<float**> result;
	std::vector<float**> error;
	for (unsigned i{ 0 }; i < vectorSize; ++i) { // tworzy potrzebne macierze
		result.push_back(matrix_operations::createMatrix(matrixSize, matrixSize));
		error.push_back(matrix_operations::createMatrix(matrixSize, matrixSize));
	}
	result_3D.push_back(result);
	error_3D.push_back(error);
}




