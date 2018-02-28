#include "stdafx.h"
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <math.h>
#include <fstream>
#include <string>
#include "fcnn.h"

using namespace std;

string  fileTrainImageMNIST = "mnist/train-images.idx3-ubyte";
string  fileTrainLabelsMNIST = "mnist/train-labels.idx1-ubyte";
string  fileTestImageMNIST = "mnist/t10k-images.idx3-ubyte";
string  fileTestLabelsMNIST = "mnist/t10k-labels.idx1-ubyte";

int width = 28, height = 28;
int numberTrainImage = 60000;
int numberTestImage = 10000;

int numberInput = width * height + 1;
int numberOutput = 10;

void readSetImage(string fileName, double **data, int n) {
	ifstream file(fileName, ios::binary);

	char numer;
	for (int i = 0; i < 16; i++) {
		file.read(&numer, sizeof(char));
	}

	for (int i = 0; i < n; i++) {
		data[i][0] = 1;
		int k = 1;
		for (int r = 0; r < 28; r++) {
			for (int c = 0; c < 28; c++) {
				unsigned char tmp = 0;
				file.read((char*)&tmp, sizeof(unsigned char));
				data[i][k] = (double)tmp / 255.0;
				k++;
			}
		}
	}
}

void readSetLabel(string filename, double *label, int n) {
	ifstream file(filename, ios::binary);

	char numer;
	for (int i = 0; i < 8; i++) {
		file.read(&numer, sizeof(char));
	}

	for (int i = 0; i < n; i++) {
		unsigned char tmp = 0;
		file.read((char*)&tmp, sizeof(unsigned char));
		label[i] = (double)tmp;
	}
}

int main(int argc, char* argv[])
{
	int numberHidden;
	int numberEpochs;
	double learningRate;
	double errorCrossEntropy;

	double **trainData = new double*[numberTrainImage];
	for (int i = 0; i < numberTrainImage; i++)
		trainData[i] = new double[numberInput];
	readSetImage(fileTrainImageMNIST, trainData, numberTrainImage);

	double *trainLabel = new double[numberTrainImage];
	readSetLabel(fileTrainLabelsMNIST, trainLabel, numberTrainImage);

	double **testData = new double*[numberTestImage];
	for (int i = 0; i < numberTestImage; i++)
		testData[i] = new double[numberInput];
	readSetImage(fileTestImageMNIST, testData, numberTestImage);

	double *testLabel = new double[numberTestImage];
	readSetLabel(fileTestLabelsMNIST, testLabel, numberTestImage);





	numberEpochs = 1;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 100;

	printf("\n Run training algorithm ... \n \n");
	fcnn *network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	double Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;



	numberEpochs = 1;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 100;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;




	numberEpochs = 1;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 100;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;










	numberEpochs = 10;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 100;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;

	numberEpochs = 10;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 150;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;


	numberEpochs = 10;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 200;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;




	numberEpochs = 20;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 100;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;

	numberEpochs = 20;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 150;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;


	numberEpochs = 20;
	learningRate = 0.01;
	errorCrossEntropy = 0.001;
	numberHidden = 200;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;






	numberEpochs = 20;
	learningRate = 0.005;
	errorCrossEntropy = 0.001;
	numberHidden = 100;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;

	numberEpochs = 20;
	learningRate = 0.005;
	errorCrossEntropy = 0.001;
	numberHidden = 150;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;


	numberEpochs = 20;
	learningRate = 0.005;
	errorCrossEntropy = 0.001;
	numberHidden = 200;

	printf("\n Run training algorithm ... \n \n");
	network = new fcnn(numberInput, numberHidden, numberOutput);
	network->train(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	Accuracy = network->calculateAccuracy(trainData, trainLabel, numberTrainImage);
	printf("Accuracy train = %f \n", Accuracy);

	Accuracy = network->calculateAccuracy(testData, testLabel, numberTestImage);
	printf("Accuracy test = %f \n", Accuracy);
	delete network;



	for (int i = 0; i < numberTrainImage; i++)
		delete[] trainData[i];
	delete[] trainData;

	for (int i = 0; i < numberTestImage; i++)
		delete[] testData[i];
	delete[] testData;

	delete[] trainLabel;
	delete[] testLabel;

	printf("runtime = %f min \n", (clock() / 1000.0) / 60);

	return 0;
}