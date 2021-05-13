#pragma once
#include <vector>
class conv3Din
{
private:
	unsigned kernelSize;
	unsigned kernelNumber;
	unsigned errorSize;
	unsigned matrixSize;
	std::vector<float**> matrix_in;
	std::vector<float**> out;
	std::vector<float**> error;
	std::vector<float**> kernels;
	std::vector<float**> batch;
public:
	conv3Din(unsigned _kernelSize, unsigned _kernelNumber, unsigned _matrixSize, unsigned _errorSize, std::vector<float**>& _matrix_in, std::vector<float**>& _out, std::vector<float**>& _error);
	~conv3Din();
	void feed_forward();
	void back_propagation();
};

