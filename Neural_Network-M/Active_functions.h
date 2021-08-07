#pragma once
#include <functional>

class Active_functions
{
private:
	std::function<void (float*, float*, unsigned size)> functions[2];
	unsigned* answer;
	unsigned size;
	float* result[2];
	float* deriative;
	std::vector<float**>* vector;
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
	Active_functions(unsigned inSize, float* resultIN, float* resultOUT, float* deriativeOUT, unsigned* answer, Active_fun _function);
	Active_functions(unsigned inSize, std::vector<float**>* _vector, Active_fun _function);
	void feed_forward();
	void feed_forwardM();
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

