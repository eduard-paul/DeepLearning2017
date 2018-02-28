#include "stdafx.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include "fcnn.h"

using namespace std;

double deriviateHyperbolicTangent(double valueTanh) {
	return (1 - valueTanh) * (1 + valueTanh);
}

double* softmax(double *g, int numberNeurons) {
	double* valueFunction = new double[numberNeurons];
	double sumExp = 0;

	for (int m = 0; m < numberNeurons; m++) {
		sumExp += exp(g[m]);
	}

	for (int j = 0; j < numberNeurons; j++) {
		valueFunction[j] = exp(g[j]) / sumExp;
	}

	return valueFunction;
}

fcnn::fcnn(int _numberInputNeurons, int _numberHiddenNeurons, int _numberOutputNeurons) {
	numberInputNeurons = _numberInputNeurons;
	numberHiddenNeurons = _numberHiddenNeurons;
	numberOutputNeurons = _numberOutputNeurons;

	inputSignal = new double[numberInputNeurons];
	expectedSignal = new double[numberOutputNeurons];

	hiddenSignal = new double[numberHiddenNeurons];
	outputsSignal = new double[numberOutputNeurons];

	weights1 = new double*[numberInputNeurons];
	for (int i = 0; i < numberInputNeurons; i++)
		weights1[i] = new double[numberHiddenNeurons];

	weights2 = new double*[numberHiddenNeurons];
	for (int i = 0; i < numberHiddenNeurons; i++)
		weights2[i] = new double[numberOutputNeurons];

	srand(time(NULL));

	for (int i = 0; i < numberInputNeurons; i++)
		for (int j = 0; j < numberHiddenNeurons; j++)
			weights1[i][j] = -0.5 + (double)rand() / RAND_MAX;

	for (int i = 0; i < numberHiddenNeurons; i++)
		for (int j = 0; j < numberOutputNeurons; j++)
			weights2[i][j] = -0.5 + (double)rand() / RAND_MAX;
}

fcnn::~fcnn() {
	delete[] inputSignal;
	delete[] hiddenSignal;
	delete[] outputsSignal;
	delete[] expectedSignal;

	for (int i = 0; i < numberInputNeurons; i++)
		delete[] weights1[i];
	delete[] weights1;

	for (int i = 0; i < numberHiddenNeurons; i++)
		delete[] weights2[i];
	delete[] weights2;
}

void fcnn::calculateOutputs() {
	double *f = new double[numberHiddenNeurons];

	for (int s = 0; s < numberHiddenNeurons; s++) {
		f[s] = 0;
		for (int i = 0; i < numberInputNeurons; i++) {
			f[s] += weights1[i][s] * inputSignal[i];
		}
		hiddenSignal[s] = tanh(f[s]);
	}
	hiddenSignal[0] = 1;


	double *g = new double[numberOutputNeurons];

	for (int j = 0; j < numberOutputNeurons; j++) {
		g[j] = 0;
		for (int s = 0; s < numberHiddenNeurons; s++) {
			g[j] += weights2[s][j] * hiddenSignal[s];
		}
	}

	outputsSignal = softmax(g, numberOutputNeurons);

	delete[] f;
	delete[] g;
}

void fcnn::calculateGradient(double **gradientWeightsLayer1, double **gradientWeightsLayer2) {
	double *sigmaLayer2 = new double[numberOutputNeurons];

	for (int s = 0; s < numberHiddenNeurons; s++) {
		for (int j = 0; j < numberOutputNeurons; j++) {
			sigmaLayer2[j] = outputsSignal[j] - expectedSignal[j];
			gradientWeightsLayer2[s][j] = sigmaLayer2[j] * hiddenSignal[s];
		}
	}

	double *dActFuncHiddenLayer = new double[numberHiddenNeurons];
	for (int s = 0; s < numberHiddenNeurons; s++) {
		dActFuncHiddenLayer[s] = 1 - (hiddenSignal[s] * hiddenSignal[s]);
	}

	double *summa = new double[numberHiddenNeurons];
	for (int s = 0; s < numberHiddenNeurons; s++) {
		summa[s] = 0;
		for (int j = 0; j < numberOutputNeurons; j++) {
			summa[s] += sigmaLayer2[j] * weights2[s][j];
		}
	}

	for (int i = 0; i < numberInputNeurons; i++) {
		for (int s = 0; s < numberHiddenNeurons; s++) {
			gradientWeightsLayer1[i][s] = dActFuncHiddenLayer[s] * summa[s] * inputSignal[i];
		}
	}

	delete[] sigmaLayer2;
	delete[] summa;
	delete[] dActFuncHiddenLayer;
}

void fcnn::correctWeights(double **gradientWeightsLayer1, double **gradientWeightsLayer2, double learningRate) {

	for (int i = 0; i < numberInputNeurons; i++) {
		for (int s = 0; s < numberHiddenNeurons; s++) {
			weights1[i][s] -= learningRate * gradientWeightsLayer1[i][s];
		}
	}

	for (int s = 0; s < numberHiddenNeurons; s++) {
		for (int j = 0; j < numberOutputNeurons; j++) {
			weights2[s][j] -= learningRate * gradientWeightsLayer2[s][j];
		}
	}
}

void fcnn::backPropagation(double learningRate) {

	double **gradientWeightsLayer1, **gradientWeightsLayer2;
	gradientWeightsLayer1 = new double*[numberInputNeurons];
	for (int i = 0; i < numberInputNeurons; i++)
		gradientWeightsLayer1[i] = new double[numberHiddenNeurons];

	gradientWeightsLayer2 = new double*[numberHiddenNeurons];
	for (int s = 0; s < numberHiddenNeurons; s++)
		gradientWeightsLayer2[s] = new double[numberOutputNeurons];

	calculateOutputs();
	calculateGradient(gradientWeightsLayer1, gradientWeightsLayer2);
	correctWeights(gradientWeightsLayer1, gradientWeightsLayer2, learningRate);

	for (int i = 0; i < numberInputNeurons; i++)
		delete[] gradientWeightsLayer1[i];
	delete[] gradientWeightsLayer1;

	for (int i = 0; i < numberHiddenNeurons; i++)
		delete[] gradientWeightsLayer2[i];
	delete[] gradientWeightsLayer2;
}

double fcnn::crossEntropy(double **data, double *labels, int N) {
	double crossEntropy = 0;

	for (int image = 0; image < N; image++) {
		for (int i = 0; i < numberInputNeurons; i++) {
			inputSignal[i] = data[image][i];
		}

		for (int j = 0; j < numberOutputNeurons; j++) {
			expectedSignal[j] = 0;
		}
		expectedSignal[(int)labels[image]] = 1;

		calculateOutputs();

		for (int j = 0; j < numberOutputNeurons; j++) {
			crossEntropy += expectedSignal[j] * log(outputsSignal[j]);
		}
	}

	crossEntropy = -1 * crossEntropy / N;

	return crossEntropy;

}

void fcnn::shuffleSamples(int *order, int size) {
	for (int i = 0; i < size; i++) {
		order[i] = i;
	}

	int rand1, rand2, tmp;
	for (int i = 0; i < size / 2; i++) {
		rand1 = (int)__min(size - 1, (int)(((double)rand() / RAND_MAX) * size));
		rand2 = (int)__min(size - 1, (int)(((double)rand() / RAND_MAX) * size));
		tmp = order[rand1];
		order[rand1] = order[rand2];
		order[rand2] = tmp;
	}
}

void fcnn::train(double **data, double *labels, int size, int numberEpochs, double learningRate, double errorCrossEntropy) {

	double currentCrossEntropy = 0;
	int numberImage = 0;

	int *order = new int[size];

	for (int epoch = 0; epoch < numberEpochs; epoch++) {
		cout << "Epoch No: " << epoch << " progress: ";

		shuffleSamples(order, size);

		for (int image = 0; image < size; image++) {
			if (image % (size / 10) == 0) cout << (100 * image / size) << "%, ";
			numberImage = order[image];
			for (int i = 0; i < numberInputNeurons; i++) {
				inputSignal[i] = data[numberImage][i];
			}

			for (int j = 0; j < numberOutputNeurons; j++) {
				expectedSignal[j] = 0;
			}
			expectedSignal[(int)labels[numberImage]] = 1;

			backPropagation(learningRate);
		}

		currentCrossEntropy = crossEntropy(data, labels, size);
		cout << endl << "CrossEntropy: " << currentCrossEntropy << endl;

		if (currentCrossEntropy < errorCrossEntropy) {
			break;
		}
	}

	delete[] order;
}

double fcnn::calculateAccuracy(double **data, double *label, int numberImage) {
	double Accuracy = 0;
	int truePositive = 0, falsePositive = 0;
	int maxIndex;

	for (int image = 0; image < numberImage; image++) {
		for (int i = 0; i < numberInputNeurons; i++) {
			inputSignal[i] = data[image][i];
		}

		for (int j = 0; j < numberOutputNeurons; j++) {
			expectedSignal[j] = 0;
		}
		expectedSignal[(int)label[image]] = 1;

		calculateOutputs();

		maxIndex = 0;
		for (int j = 0; j < numberOutputNeurons; j++) {
			if (outputsSignal[j] > outputsSignal[maxIndex]) {
				maxIndex = j;
			}
		}

		if (expectedSignal[maxIndex] == 1.0) {
			truePositive++;
		}
		else {
			falsePositive++;
		}
	}

	Accuracy = (double)truePositive / (double)(truePositive + falsePositive);

	return Accuracy;
}


