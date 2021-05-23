#include "Initializator.h"

float Initializator::He_ini(unsigned numberInputs)
{
	return 2.0f / numberInputs;
}

float Initializator::Xavier_ini(unsigned numberInputs, unsigned numberOutput)
{
	return 1.0f/(numberInputs+numberOutput);
}
