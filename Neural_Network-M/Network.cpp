#include "Network.h"

void Network::add3Dconv(unsigned kernelNumber, unsigned kernelSize)
{
	Layers.push_back(kernelNumber, kernelSize+)
}

void Network::addFullyCon(unsigned neuronNumber, unsigned inNumber)
{
	result_fullyCon.push_back(new float[neuronNumber]);
	result_deriative.push_back(new float[neuronNumber]);
	dercost_fullyCon.push_back(new float[neuronNumber]);
	Layers.push_back(new fully_connected(neuronNumber, inNumber, result_fullyCon.back(), result_funfullyCon.back(), result_deriative.back(), dercost_fullyCon.back()));
}


