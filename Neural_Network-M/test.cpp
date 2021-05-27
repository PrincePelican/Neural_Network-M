#include <iostream>
#include "Network.h"
#include "matrix_operations.h"
#include <time.h>
#include <fstream>

using namespace std;

class Matrix {
public:
    float** tablica;
    Matrix() {
        tablica = new float* [28];
        for (unsigned i = 0; i < 28; ++i)
            tablica[i] = new float[28];
    }
    ~Matrix() {
        for (unsigned i = 0; i < 28; ++i)
            delete tablica[i];
        delete tablica;
    }
};

void read_mnist_images(string full_path, vector<float**>& images) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0, number_of_images = 0, image_size = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;
        for (unsigned i = 0; i < number_of_images; ++i) {
            images.push_back(matrix_operations::createMatrix(n_rows, n_cols));
        }
        unsigned char* _dataset = new unsigned char[number_of_images];
        for (int x = 0; x < number_of_images; x++) {
            file.read((char*)_dataset, image_size);
            for (unsigned i = 0; i < n_rows; ++i) {
                for (unsigned j = 0; j < n_cols; ++j) {
                    images[x][i][j] = (float)_dataset[i * n_cols + j] / (float)255;
                }
            }
        }
        delete _dataset;
    }
    else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

void read_mnist_labels(string full_path, vector<unsigned int>& labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };


    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0, number_of_labels;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);
        labels.resize(number_of_labels);

        unsigned char* _dataset = new unsigned char[number_of_labels];

        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
            labels[i] = (int)_dataset[i];
        }
        delete _dataset;
    }
    else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}


int main() {
    vector<float**> images;
    vector<float**> test;
    vector<unsigned int> labels;
    vector<unsigned int> labels_test;

    read_mnist_images("train-images.idx3-ubyte", images);
    read_mnist_labels("train-labels.idx1-ubyte", labels);
    read_mnist_images("t10k-images.idx3-ubyte", test);
    read_mnist_labels("t10k-labels.idx1-ubyte", labels_test);


	Network A;
	A.add2Dconv(6, 3, 28);
	A.addPooling(2);
	A.add3Dconv(6, 3); // problem z wymiarami podczas tworzenia tablic error in out Ÿle przypisowane s¹ iloœæ filtrów 
	A.addPooling(2, true);
	A.addFullyCon(Active_functions::Active_fun::RELU, 150);
	A.addFullyCon(Active_functions::Active_fun::SOFTMAX, 10);
	A.initializatiion(Initializator::He);
    A.changeLearnRate(0.001);
    A.Learn(&images, &labels);
    A.Predict(&test, &labels_test);
    A.Learn(&images, &labels);
    A.Predict(&test, &labels_test);
    A.Learn(&images, &labels);
    A.Predict(&test, &labels_test);
	return 0;
}