#include "matrix_operations.h"
#include <iostream>

void matrix_operations::dot_product(float*& out, float**& matrix_in, float*& vector_in, unsigned matrixSizeY, unsigned matrixSizeX)
{
	for (unsigned i{ 0 }; i < matrixSizeY; ++i) {
		for (unsigned j{ 0 }; j < matrixSizeX; ++j) {
			out[i] += matrix_in[i][j] * vector_in[j];
		}
	}
}

void matrix_operations::dot_productB(float*& out, float**& matrix_in, float*& vector_in, unsigned matrixSizeY, unsigned matrixSizeX)
{
	for (unsigned i{ 0 }; i < matrixSizeY; ++i) {
		for (unsigned j{ 0 }; j < matrixSizeX; ++j) {
			out[j] += matrix_in[i][j] * vector_in[i];
		}
	}
}

void matrix_operations::multiply(float*& out, float*& vector1, float*& vector2, unsigned size)
{
	for (unsigned i{ 0 }; i < size; ++i) {
		out[i] = vector1[i] * vector2[i];
	}
}

void matrix_operations::multiply(float**& out, unsigned sizeY, unsigned sizeX, float multiplier)
{
	for (unsigned i{ 0 }; i < sizeY; ++i) {
		for (unsigned j{ 0 }; j < sizeX; ++j) {
			out[i][j] *= multiplier;
		}
	}
}

void matrix_operations::multiply(float*& out, unsigned size, float multiplier)
{
	for (unsigned i{ 0 }; i < size; ++i) {
		out[i] *= multiplier;
	}
}

void matrix_operations::add(float**& out, float**& matrix_in, unsigned sizeY, unsigned sizeX)
{
	for (unsigned i{ 0 }; i < sizeY; ++i) {
		for (unsigned j{ 0 }; j < sizeX; ++j) {
			out[i][j] += matrix_in[i][j];
		}
	}
}

void matrix_operations::add(float*& out, float*& matrix_in, unsigned size)
{
	for (unsigned i{ 0 }; i < size; ++i)
		out[i] += matrix_in[i];
}

void matrix_operations::subtract(float**& out, float**& matrix_in, unsigned sizeY, unsigned sizeX)
{
	for (unsigned i{ 0 }; i < sizeY; ++i) {
		for (unsigned j{ 0 }; j < sizeX; ++j) {
			out[i][j] -= matrix_in[i][j];
		}
	}
}

void matrix_operations::subtract(float* out, float* matrix_in, float* subtractor, unsigned size)
{
	for (unsigned i{ 0 }; i < size; ++i) {
		out[i] = matrix_in[i] - subtractor[i];
	}
}

void matrix_operations::subtract(float* out, float* subtractor, unsigned size)
{
	for (unsigned i{ 0 }; i < size; ++i) {
		out[i] -= subtractor[i];
	}
}

void matrix_operations::add(float**& out, std::vector<float**>& matrix_in, unsigned matrixSize)
{
	for (unsigned i{ 0 }; i < matrixSize; ++i) {
		for (unsigned j{ 0 }; j < matrixSize; ++j) {
			for (float**& in : matrix_in) {
				out[i][j] += in[i][j];
			}
		}
	}
}

float** matrix_operations::createMatrix(unsigned sizeY, unsigned sizeX)
{
	float** newMatrix = new float*[sizeY];
	for (unsigned i{ 0 }; i < sizeY; ++i)
		newMatrix[i] = new float[sizeX] {0};
	return newMatrix;
}

void matrix_operations::clearMatrix(float**& matrix, unsigned sizeY, unsigned sizeX)
{
	for (unsigned i{ 0 }; i < sizeY; ++i) delete matrix[i];
	delete[] matrix;
}

void matrix_operations::showMatrix(float**& matrix, unsigned sizeY, unsigned sizeX)
{
	std::cout << std::endl << std::endl;
	for (unsigned i{ 0 }; i < sizeY; ++i) {
		for (unsigned j{ 0 }; j < sizeX; ++j) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void matrix_operations::showVector(float*& matrix, unsigned size)
{
	std::cout << std::endl << std::endl;
	for (unsigned i{ 0 }; i < size; ++i)
		std::cout << matrix[i] << " ";
}

void matrix_operations::ResetMem(float**& matrix, unsigned sizeY, unsigned sizeX)
{
	for (unsigned i{ 0 }; i < sizeY; ++i)
		memset(matrix[i], 0, sizeX * sizeof(float));
}

void matrix_operations::ResetMem(float*& matrix, unsigned size)
{
	memset(matrix, 0, size * sizeof(float));
}

void matrix_operations::pooling(float** in, float** out, unsigned matrixSize, unsigned pooling_size)
{
	for (unsigned i{ 0 }; i < matrixSize - (pooling_size - 1); i = i + pooling_size) {	//przechodzi przez macierz i robi avg pooling
		for (unsigned j{ 0 }; j < matrixSize - (pooling_size - 1); j = j + pooling_size)
		{
			out[i / pooling_size][j / pooling_size] = avg_feed_forward(in, i, j, pooling_size, pooling_size * pooling_size);
		}
	}
}

void matrix_operations::pool_back(float** in, float** out, unsigned matrixSize, unsigned pooling_size)
{
	float divider = pooling_size * pooling_size;
	unsigned outSize = matrixSize * pooling_size;
	for (unsigned i{ 0 }; i < matrixSize; ++i) {
		for (unsigned j{ 0 }; j < matrixSize; ++j) {
			avg_back_prop(out, in, i, j, pooling_size, divider);
		}
	}
}

float matrix_operations::avg_feed_forward(float** out, unsigned y, unsigned x, unsigned pooling_size,float divider)
{
	float result = 0;
	for (unsigned i{ 0 }; i < pooling_size; ++i) {
		for (unsigned j{ 0 }; j < pooling_size; ++j) {
			result += out[y + i][x + i];
		}
	}
	return result / divider;
}

void matrix_operations::avg_back_prop(float** out, float** in, unsigned y, unsigned x, unsigned pooling_size, float divider)
{
	float result = in[y][x] / divider;
	for (unsigned i{ 0 }; i < pooling_size; ++i) {
		for (unsigned j{ 0 }; j < pooling_size; ++j) {
			out[y*2 + i][x*2 + j] = result;
		}
	}
}



void matrix_operations::assignTo3D(float* in, std::vector<float**>* out, unsigned matrixSize)
{
	unsigned counter = 0;
	for (float**& matrix : (*out)) {
		for (unsigned i{ 0 }; i < matrixSize; ++i) {
			for (unsigned j{ 0 }; j < matrixSize; ++j) {
				matrix[i][j] = in[counter];
				counter++;
			}
		}
	}
}

void matrix_operations::assingToflatten(std::vector<float**> *in, float* out, unsigned matrixSize)
{
	unsigned counter = 0;
	for (float**& matrix : (*in)) {
		for (unsigned i{ 0 }; i < matrixSize; ++i) {
			for (unsigned j{ 0 }; j < matrixSize; ++j) {
				out[counter] = matrix[i][j];
				counter++;
			}
		}
	}
}

void matrix_operations::Conv(float**& matrix, float**& kernel, float**& out, unsigned matrixSize, unsigned kernelSize)
{
	for (unsigned i{ 0 }; i < matrixSize - (kernelSize - 1); ++i) {
		for (unsigned j{ 0 }; j < matrixSize - (kernelSize - 1); ++j) {
			out[i][j] = convOperation(matrix, kernel, kernelSize, i, j);
		}
	}
}


float matrix_operations::convOperation(float**& matrix, float**& kernel, unsigned kernelSize, unsigned y, unsigned x)
{
	float result{0};
	for (unsigned i{ 0 }; i < kernelSize; ++i) {
		for (unsigned j{ 0 }; j < kernelSize; ++j) {
			result += matrix[y + i][x + j] * kernel[i][j];
		}
	}
	return result;
}

void matrix_operations::dot_product(float**& out, float*& vector1, float*& vector2, unsigned vector1Size, unsigned vector2Size)
{
	for (unsigned i{ 0 }; i < vector1Size; ++i) {
		for (unsigned j{ 0 }; j < vector2Size; ++j) {
			out[i][j] = vector1[i] * vector2[j];
		}
	}
}

void matrix_operations::transpose(float**& matrix, unsigned size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0, k = size - 1; j < k; j++, k--)
			std::swap(matrix[j][i], matrix[k][i]);
	}
}

void matrix_operations::rotate180(float**& matrix, unsigned size)
{
	transpose(matrix, size);
	reverseColumns(matrix, size);
	transpose(matrix, size);
	reverseColumns(matrix, size);
}

void matrix_operations::reverseColumns(float**& matrix, unsigned size)
{
	for (int i = 0; i < size; i++)
		for (int j = i; j < size; j++)
			std::swap(matrix[i][j], matrix[j][i]);
}

unsigned matrix_operations::chooseMax(float*& matrix, unsigned size)
{
	float max = matrix[0];
	unsigned n_max = 0;
	for (int i = 0; i < size; i++) {
		if (max < matrix[i]) {
			max = matrix[i];
			n_max = i;
		}
	}
	return n_max;
}
