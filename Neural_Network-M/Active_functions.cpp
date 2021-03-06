#include "Active_functions.h"
#include <iostream>

Active_functions::Active_functions(unsigned inSize, float* resultIN, float* resultOUT, float* deriativeOUT, unsigned* answer, Active_fun _function)
{
	this->result[0] = resultIN;
	this->result[1] = resultOUT;
	this->deriative = deriativeOUT;
	this->size = inSize;
	this->answer = answer;

	set_active_fun(_function);
}

Active_functions::Active_functions(unsigned inSize, std::vector<float**>* _vector, Active_fun _function)
{
	this->vector = _vector;
	this->size = inSize;

	set_active_fun(_function);
}

void Active_functions::feed_forward()
{
	functions[0](result[0], result[1], size);
}

void Active_functions::feed_forwardM()
{
	for (float** in : *vector) {
		for (unsigned i{ 0 }; i < size; ++i)
			functions[0](in[i], in[i], size);
	}
}

void Active_functions::deriative_out()
{
	functions[1](result[1], deriative, size);
}

void Active_functions::set_active_fun(Active_fun fun)
{
	switch (fun) {
	case Active_fun::TANH:
		functions[0] = std::bind(&Active_functions::tanh, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		functions[1] = std::bind(&Active_functions::tanh_der, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		break;
	case Active_fun::SIGMOID:
		functions[0] = std::bind(&Active_functions::sigmoid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		functions[1] = std::bind(&Active_functions::sigmoid_der, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		break;
	case Active_fun::ARCTAN:
		functions[0] = std::bind(&Active_functions::arctan, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		functions[1] = std::bind(&Active_functions::arctan_der, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		break;
	case Active_fun::RELU:
		functions[0] = std::bind(&Active_functions::relu, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		functions[1] = std::bind(&Active_functions::relu_der, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		break;
	case Active_fun::SOFTPLUS:
		functions[0] = std::bind(&Active_functions::softplus, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		functions[1] = std::bind(&Active_functions::softplus_der, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		break;
	case Active_fun::SOFTMAX:
		functions[0] = std::bind(&Active_functions::softmax, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		functions[1] = std::bind(&Active_functions::softmax_der, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
		break;

	}
}


void Active_functions::tanh(float* in, float* out, unsigned size)
{
	for (unsigned i{ 0 }; i < size; ++i)
		out[i] = std::tanh(in[i]);
}

	void Active_functions::sigmoid(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = 1 / (1 - std::exp(in[i]));
	}

	void Active_functions::relu(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = in[i] < 0 ? 0.01 * in[i] : in[i];
	}
	void Active_functions::arctan(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = atan(in[i]);
	}

	void Active_functions::softplus(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = std::log(1 + std::exp(in[i]));
			
	} 

	void Active_functions::softmax(float* in, float* out, unsigned size)
	{
		float sum = 0;
		float max = in[0];
		for (unsigned i{ 0 }; i < size; ++i) {
			if (in[i] > max)
				max = in[i];
		}
		for (unsigned i{ 0 }; i < size; ++i) {
			sum += std::exp(in[i]-max);
		}
		for (unsigned i{ 0 }; i < size; ++i) {
			out[i] = std::exp(in[i]-max)/ sum;
		}
	}

	void Active_functions::arctan_der(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = 1 / 1 + pow(in[i],2);
	}

	void Active_functions::sigmoid_der(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = 1 / (1 - std::exp(in[i]));
	}

	void Active_functions::softplus_der(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = in[i] * (1 - in[i]);
	}

	void Active_functions::softmax_der(float* in, float* out, unsigned size) 
	{
		for (unsigned i{ 0 }; i < size; ++i) {
			if (i == *answer) {
				out[i] = in[i] * (1 - in[i]);
			}
			else{
				out[i] = in[*answer] * -in[i];
			}
			
		}
	}

	void Active_functions::relu_der(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = in[i] > 0 ? 1 : 0.01 * in[i];
	}

	void Active_functions::tanh_der(float* in, float* out, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			out[i] = 1 - pow(std::tanh(in[i]), 2);
	}
