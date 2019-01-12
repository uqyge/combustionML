#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

// using namespace std;

class TrainingData
{
  public:
	TrainingData(const std::string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
	void getTopology(std::vector<unsigned> &topology);

	// Returns the number of input values read from the file:
	unsigned getNextInputs(std::vector<double> &inputVals);
	unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

  private:
	std::ifstream m_trainingDataFile;
};

void TrainingData::getTopology(std::vector<unsigned> &topology)
{
	std::string line;
	std::string label;

	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const std::string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
	inputVals.clear();

	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
  public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);

  private:
	static double eta;   // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double transferFunction(double x);
	// randomWeight: 0 - 1
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned m_myIndex;
};

double Neuron::eta = 0.15;  // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	return tanh(x);
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
			   prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}
// ****************** class Net ******************
class Net
{
  public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void getResults(std::vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

  private:
	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(std::vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::feedForward(const std::vector<double> &inputVals)
{
	// Check the num of inputVals euqal to neuronnum expect bias
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}
Net::Net(const std::vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.push_back(Layer());
		// numOutputs of layer[i] is the numInputs of layer[i+1]
		// numOutputs of last layer is 0
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer:
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "Mad a Neuron!" << std::endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void showVectorVals(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}
void showVectorVals(std::string label, std::vector<unsigned> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

int main()
{
	TrainingData trainData("trainingData.txt");
	//e.g., {3, 2, 1 }
	std::vector<unsigned> topology;
	// topology.push_back(2);
	// topology.push_back(4);
	// topology.push_back(1);

	trainData.getTopology(topology);
	Net myNet(topology);
	showVectorVals(": Topology :", topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << std::endl
				  << "Pass" << trainingPass << "\n";

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());
	}

	std::cout << std::endl
			  << "Done" << std::endl;
}
