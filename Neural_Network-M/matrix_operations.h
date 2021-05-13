#pragma once
#include <vector>
class matrix_operations
{
public:
	static void dot_product(float*& out, float**& matrix_in, float*& vector_in, unsigned matrixSizeY, unsigned matrixSizeX);
	static void dot_product(float**& out, float*& vector1, float*& vector2, unsigned vector1Size, unsigned vector2Size);
	static void multiply(float*& out, float*& vector1, float*& vector2, unsigned size);
	static void add(float**& out, float**& matrix_in, unsigned sizeY, unsigned sizeX);
	static void subtract(float**& out, float**& matrix_in, unsigned sizeY, unsigned sizeX);
	static void add(float**& out, std::vector<float**>& matrix_in, unsigned matrixSize);
	///
	static float** createMatrix(unsigned sizeY, unsigned sizeX);
	static void clearMatrix(float**& matrix, unsigned sizeY, unsigned sizeX);
	static void showMatrix(float**& matrix, unsigned sizeY, unsigned sizeX);
	static void ResetMem(float**& matrix, unsigned sizeY, unsigned sizeX);
	/// conv operation
	static void Conv(float**& matrix,float**& kernel ,float**& out, unsigned matrixSize, unsigned kernelSize);
private:
	static float convOperation(float**& matrix, float**& kernel, unsigned kernelSize, unsigned y, unsigned x);
};

