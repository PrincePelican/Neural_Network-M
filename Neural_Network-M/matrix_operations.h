#pragma once
#include <vector>
class matrix_operations
{
public:
	static void dot_product(float*& out, float**& matrix_in, float*& vector_in, unsigned matrixSizeY, unsigned matrixSizeX);
	static void dot_productB(float*& out, float**& matrix_in, float*& vector_in, unsigned matrixSizeY, unsigned matrixSizeX);
	static void dot_product(float**& out, float*& vector1, float*& vector2, unsigned vector1Size, unsigned vector2Size);
	static void multiply(float*& out, float*& vector1, float*& vector2, unsigned size);
	static void multiply(float**& out, unsigned sizeY, unsigned sizeX, float multiplier);
	static void add(float**& out, float**& matrix_in, unsigned sizeY, unsigned sizeX);
	static void add(float**& out, std::vector<float**>& matrix_in, unsigned matrixSize);
	static void subtract(float**& out, float**& matrix_in, unsigned sizeY, unsigned sizeX);
	static void subtract(float* out, float* matrix_in, float* subtractor, unsigned size);
	static void rotate180(float**& matrix, unsigned size);
	static void transpose(float**& matrix, unsigned size);
	static void reverseColumns(float**& matrix, unsigned size);
	static unsigned chooseMax(float*& matrix, unsigned size);
	///
	static float** createMatrix(unsigned sizeY, unsigned sizeX);
	static void clearMatrix(float**& matrix, unsigned sizeY, unsigned sizeX);
	static void showMatrix(float**& matrix, unsigned sizeY, unsigned sizeX);
	static void showVector(float*& matrix, unsigned size);
	static void ResetMem(float**& matrix, unsigned sizeY, unsigned sizeX);
	static void ResetMem(float*& matrix, unsigned size);
	/// conv operation
	static void pooling(float** in, float** out, unsigned matrixSize, unsigned pooling_size);
	static void pool_back(float** in, float** out, unsigned matrixSize, unsigned pooling_size);
	static void assignTo3D(float* in, std::vector<float**>* out, unsigned matrixSize);
	static void assingToflatten(std::vector<float**>* in, float* out, unsigned matrixSize);
	static void Conv(float**& matrix,float**& kernel ,float**& out, unsigned matrixSize, unsigned kernelSize);
private:
	static float convOperation(float**& matrix, float**& kernel, unsigned kernelSize, unsigned y, unsigned x);
	static float avg_feed_forward(float** out, unsigned y, unsigned x, unsigned pooling_size, float divider);
	static void avg_back_prop(float** out, float** in, unsigned y, unsigned x, unsigned pooling_size, float divider);
};

