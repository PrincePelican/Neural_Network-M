#include "Active_functions.h"

void Active_functions::set_active_fun(Active_fun fun)
{
	std::function<float*(float*, unsigned size)> f;
	switch (fun) {
	case Active_fun::TANH:
		f = std::bind(&Active_functions::tanh, this, std::placeholders::_1, std::placeholders::_2);
		active_functions.push_back(f);
		break;
	case Active_fun::SIGMOID:
		f = std::bind(&Active_functions::sigmoid, this, std::placeholders::_1, std::placeholders::_2);
		active_functions.push_back(f);
		break;
	case Active_fun::ARCTAN:
		f = std::bind(&Active_functions::arctan, this, std::placeholders::_1, std::placeholders::_2);
		active_functions.push_back(f);
		break;
	case Active_fun::RELU:
		f = std::bind(&Active_functions::relu, this, std::placeholders::_1, std::placeholders::_2);
		active_functions.push_back(f);
		break;
	case Active_fun::SOFTPLUS:
		f = std::bind(&Active_functions::softplus, this, std::placeholders::_1, std::placeholders::_2);
		active_functions.push_back(f);
		break;
	case Active_fun::SOFTMAX:
		f = std::bind(&Active_functions::softmax, this, std::placeholders::_1, std::placeholders::_2);
		active_functions.push_back(f);
		break;

	}
}


float* Active_functions::tanh(float* x, unsigned size)
{
	for (unsigned i{ 0 }; i < size; ++i)
		x[i] = std::tanh(x[i]);
	return x;
}

	float* Active_functions::sigmoid(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = 1 / (1 - std::exp(x[i]));
		return x;
	}

	float* Active_functions::relu(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = x[i] < 0 ? 0.0001, 1 * x[i] : x[i];
		return x;
	}
	float* Active_functions::arctan(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			atan(x[i]);
		return x;
	}

	float* Active_functions::softplus(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = std::log(1 + std::exp(x[i]));
		return x;
	}

	float* Active_functions::softmax(float* x, unsigned size)
	{
		float sum = 0;
		float max = x[0];
		for (unsigned i{ 0 }; i < size; ++i) {
			if (x[i] > max)
				max = x[i];
		}
		for (unsigned i{ 0 }; i < size; ++i) {
			x[i] = std::exp(x[i] - max);
			sum += x[i];
		}
		for (unsigned i{ 0 }; i < size; ++i) {
			x[i] = x[i]/sum;
		}
		return x;
	}

	float* Active_functions::arctan_der(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = 1 / 1 + pow(x[i],2);
		return x;
	}

	float* Active_functions::sigmoid_der(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = 1 / (1 - std::exp(x[i]));
		return x;
	}

	float* Active_functions::softplus_der(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = x[i] * (1 - x[i]);
		return x;
	}

	float* Active_functions::softmax_der(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = x[i] * (1 - x[i]);
		return x;
	}

	float* Active_functions::relu_der(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = x[i] > 0 ? 1 : 0.0001 * x[i];
		return x;
	}

	float* Active_functions::tanh_der(float* x, unsigned size)
	{
		for (unsigned i{ 0 }; i < size; ++i)
			x[i] = 1 - pow(std::tanh(x[i]), 2);
		return x;
	}
