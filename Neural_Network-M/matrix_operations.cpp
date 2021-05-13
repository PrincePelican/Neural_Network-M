#include "matrix_operations.h"
#include <iostream>

void matrix_operations::dot_product(float*& out, float**& matrix_in, float*& vector_in, unsigned matrixSizeX, unsigned matrixSizeY)
{
	for (unsigned i{ 0 }; i < matrixSizeY; ++i) {
		for (unsigned j{ 0 }; j < matrixSizeX; ++j) {
			out[i] += matrix_in[i][j] * vector_in[j];
		}
	}
}

void matrix_operations::multiply(float*& out, float*& vector1, float*& vector2, unsigned size)
{
	for (unsigned i{ 0 }; i < size; ++i) {
		out[i] = vector1[i] * vector2[i];
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

void matrix_operations::subtract(float**& out, float**& matrix_in, unsigned sizeY, unsigned sizeX)
{
	for (unsigned i{ 0 }; i < sizeY; ++i) {
		for (unsigned j{ 0 }; j < sizeX; ++j) {
			out[i][j] -= matrix_in[i][j];
		}
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
	for (unsigned i{ 0 }; i < sizeY; ++i) {
		for (unsigned j{ 0 }; j < sizeX; ++j) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void matrix_operations::ResetMem(float**& matrix, unsigned sizeY, unsigned sizeX)
{
	for (unsigned i{ 0 }; i < sizeY; ++i)
		memset(matrix[i], 0, sizeX * sizeof(float));
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
	float result;
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