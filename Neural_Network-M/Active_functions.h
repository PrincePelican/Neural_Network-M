#pragma once
#include <functional>

class Active_functions
{
private:
	std::function<void (float*, float*, unsigned size)> functions[2];
	unsigned size;
	float* result[2];
	float* deriative;
public:
	enum class Active_fun { // klasa pomocnicza sluzaca "bezpiecznemu" wybieraniu funkcji aktywacji
		BINARY_STEP,
		SIGMOID,
		RELU,
		TANH,
		ARCTAN,
		SOFTPLUS,
		SOFTMAX
	};
	Active_functions(unsigned inSize, float* resultIN, float* resultOUT, float* deriativeOUT, Active_fun _function);
	void feed_forward();
	void deriative_out();
private:
	void set_active_fun(Active_fun fun);
	void sigmoid(float* in, float* out, unsigned size);
	void tanh(float* in, float* out, unsigned size);
	void relu(float* in, float* out, unsigned size);
	void arctan(float* in, float* out, unsigned size);
	void softplus(float* in, float* out, unsigned size);
	void softmax(float* in, float* out, unsigned size);
	void arctan_der(float* in, float* out, unsigned size);
	void softmax_der(float* in, float* out, unsigned size);
	void relu_der(float* in, float* out, unsigned size);
	void sigmoid_der(float* in, float* out, unsigned size);
	void softplus_der(float* in, float* out, unsigned size);
	void tanh_der(float* in, float* out, unsigned size);
};

