class fcnn {

private:

	int numberInputNeurons;
	int numberHiddenNeurons;
	int numberOutputNeurons;

	double *inputSignal, *hiddenSignal, *outputsSignal, *expectedSignal;
	double **weights1, **weights2;

	void calculateOutputs();
	void calculateGradient(double **gradientWeightsLayer1, double **gradientWeightsLayer2);
	void correctWeights(double **gradientWeightsLayer1, double **gradientWeightsLayer2, double learningRate);
	void backPropagation(double learningRate);
	void shuffleSamples(int *order, int size);
	double crossEntropy(double **trainData, double *trainLabel, int numberTrainImage);

public:

	fcnn(int _numberInputNeurons, int _numberHiddenNeurons, int _numberOutputNeurons);	
	double calculateAccuracy(double **data, double *label, int numberImage);
	void train(double **trainData, double *trainLabel, int numberTrainImage, int numberEpochs, double learningRate, double errorCrossEntropy);
	~fcnn();
};
