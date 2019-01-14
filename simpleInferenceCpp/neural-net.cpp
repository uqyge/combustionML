#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;

template <typename T>
std::vector<T> as_vector(ptree const &pt, ptree::key_type const &key)
{
	std::vector<T> r;
	for (auto &item : pt.get_child(key))
		r.push_back(item.second.get_value<T>());
	return r;
}

template <typename T>
void showVectorVals(std::string label, std::vector<T> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
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
	void feedForward(const Layer &prevLayer, std::string);

  private:
	static double transferFunction(double x, std::string actFunc);
	// randomWeight: 0 - 1
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned m_myIndex;
};

double Neuron::transferFunction(double x, std::string actFunc = "relu")
{
	// tanh - output range [-1.0..1.0]
	// return tanh(x);
	if (actFunc == "relu")
	{
		return std::max(0.0, x);
	}
	else if (actFunc == "linear")
	{
		return x;
	}
}

void Neuron::feedForward(const Layer &prevLayer, std::string actFunc = "relu")
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
			   prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum, actFunc);
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

  private:
	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double m_error;
};

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
	// Check the num of inputVals equal to neuronnum expect bias
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size() - 1; ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
		std::cout << layerNum << "/" << m_layers.size() << '\n';
	}
	{
		unsigned linearOutputLayer = m_layers.size() - 1;
		Layer &prevLayer = m_layers[linearOutputLayer - 1];

		for (unsigned n = 0; n < m_layers[linearOutputLayer].size() - 1; ++n)
		{
			m_layers[linearOutputLayer][n].feedForward(prevLayer, "linear");
		}
	}
}
Net::Net(const std::vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	std::cout << "topo size:" << numLayers << '\n';
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
			std::cout << "Mad a Neuron!" << neuronNum << std::endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

int main()
{
	std::ifstream myfile;
	// myfile.open("test.json");
	myfile.open("data.json");
	std::stringstream buffer;
	buffer << myfile.rdbuf();
	std::cout << buffer.str() << '\n';

	// Read json.
	ptree pt2;
	read_json(buffer, pt2);
	std::vector<std::vector<float>> layer1;
	// BOOST_FOREACH (ptree::value_type &v, pt2.get_child("l1"))
	for (auto &v : pt2.get_child("l1"))
	{
		std::vector<float> tmp;
		tmp.clear();
		for (auto &i : v.second)
		{
			tmp.push_back(i.second.get_value<float>());
		}
		layer1.push_back(tmp);
		// showVectorVals<float>("wudi:", tmp);
	}
	for (auto &i : layer1)
	{
		showVectorVals<float>("jimo:", i);
	}

	std::vector<unsigned> topology;
	topology = as_vector<unsigned>(pt2, "topology");
	showVectorVals<unsigned>(":json read topology", topology);
	Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while (trainingPass < 1)
	{
		++trainingPass;
		std::cout << "Pass:" << trainingPass << "\n";

		inputVals = as_vector<double>(pt2, "in");
		showVectorVals<double>(": Inputs :", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		showVectorVals<double>("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		targetVals = as_vector<double>(pt2, "out");
		showVectorVals<double>("Targets:", targetVals);
		assert(targetVals.size() == topology.back());
	}

	std::cout << std::endl
			  << "Done" << std::endl;
}
