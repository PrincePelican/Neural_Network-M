#pragma once
#include <functional>

class Active_functions
{
public:
	std::vector<std::function<float*(float*, unsigned size)>> active_functions;	//vektor przechowujacy funkcje aktywacji
	enum class Active_fun { // klasa pomocnicza sluzaca "bezpiecznemu" wybieraniu funkcji aktywacji
		BINARY_STEP,
		SIGMOID,
		RELU,
		TANH,
		ARCTAN,
		SOFTPLUS,
		SOFTMAX
	};
	void set_active_fun(Active_fun fun);
	float* sigmoid(float* x, unsigned size);
	float* tanh(float* x, unsigned size);
	float* relu(float* x, unsigned size);
	float* arctan(float* x, unsigned size);
	float* softplus(float* x, unsigned size);
	float* softmax(float* x, unsigned size);
	float* arctan_der(float* x, unsigned size);
	float* softmax_der(float* x, unsigned size);
	float* relu_der(float* x, unsigned size);
	float* sigmoid_der(float* x, unsigned size);
	float* softplus_der(float* x, unsigned size);
	float* tanh_der(float* x, unsigned size);
};

