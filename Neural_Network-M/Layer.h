#pragma once

#include <random>
#include <time.h>
#include "Initializator.h"

class Layer
{
public:
	virtual void feed_forward() = 0;
	virtual void back_propagation() = 0;
	virtual void initweights(Initializator::Initializators method) = 0;
	virtual void weights_update() = 0;
	virtual void changeLearnRate(float rate) = 0;
};

