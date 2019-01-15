#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <ctime>

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
std::vector<std::vector<T>> asLayerVector(ptree const &pt, ptree::key_type const &key)
{
	std::vector<std::vector<T>> r;
	for (auto &item : pt.get_child(key))
	{
		std::vector<T> tmp;
		tmp.clear();
		for (auto &connectionWeight : item.second)
			tmp.push_back(connectionWeight.second.get_value<T>());
		r.push_back(tmp);
	}
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
};

class Neuron;

typedef std::vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
  public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	Neuron(std::vector<float> connectionWeights, unsigned myIndex);
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

Neuron::Neuron(std::vector<float> connectionWeights, unsigned myIndex)
{
	for (auto &v : connectionWeights)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = v;
	}

	m_myIndex = myIndex;
}

// ****************** class Net ******************
class Net
{
  public:
	Net(const std::vector<std::vector<std::vector<float>>> &modelWeights);
	void feedForward(const std::vector<double> &inputVals);
	void getResults(std::vector<double> &resultVals) const;
	bool infer(std::vector<double> &data_in, std::vector<double> &out);

  private:
	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};

void Net::getResults(std::vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size(); ++n)
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
			// std::cout << "n:" << n << '\n';
			m_layers[layerNum][n].feedForward(prevLayer, "relu");
		}
	}
	//output layer
	{
		unsigned linearOutputLayer = m_layers.size() - 1;
		Layer &prevLayer = m_layers[linearOutputLayer - 1];
		for (auto &n : m_layers[linearOutputLayer])
		{
			n.feedForward(prevLayer, "linear");
		}
	}
}

bool Net::infer(std::vector<double> &data_in, std::vector<double> &out)
{
	std::vector<double> inputVals, resultVals;
	int trainingPass = 0;
	auto t_start = std::chrono::high_resolution_clock::now();

	while (trainingPass < 1e6)
	{
		++trainingPass;
		// std::cout << "Pass:" << trainingPass << "\n";

		inputVals = data_in;

		this->feedForward(inputVals);

		// Collect the net's actual results:
		this->getResults(resultVals);
		// showVectorVals<double>("Outputs:", resultVals);

		// assert(targetVals.size() == topology.back());
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	std::cout << "totol time is " << total << "ms.\n";
	out = resultVals;
	return 0;
}

Net::Net(const std::vector<std::vector<std::vector<float>>> &modelWeight)
{
	unsigned numLayers = modelWeight.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		std::cout << "layer:" << layerNum << '\n';
		m_layers.push_back(Layer());

		std::vector<std::vector<float>> layerWeights = modelWeight[layerNum];
		for (unsigned neuronNum = 0; neuronNum < modelWeight[layerNum].size(); ++neuronNum)
		{
			m_layers.back().push_back(Neuron(layerWeights[neuronNum], neuronNum));
			// std::cout << "Mad a Neuron! Index:" << neuronNum << std::endl;
		}
		m_layers.back().back().setOutputVal(1.0);
	}
	//output layer
	std::cout << "layer:output\n";
	m_layers.push_back(Layer());
	for (unsigned neuronNum = 0; neuronNum < modelWeight[numLayers - 1][0].size(); ++neuronNum)
	{
		m_layers.back().push_back(Neuron(0, neuronNum));
		// std::cout << "Mad a Neuron! Index:" << neuronNum << std::endl;
	}
	m_layers.back().back().setOutputVal(1.0);

	assert(m_layers.size() == (modelWeight.size() + 1));
}

int main()
{
	std::ifstream myfile;
	myfile.open("data.json");
	// myfile.open("test.json");
	std::stringstream buffer;
	buffer << myfile.rdbuf();
	// std::cout << buffer.str() << '\n';

	ptree pt2;
	read_json(buffer, pt2);

	std::vector<std::vector<std::vector<float>>> modelWeights;
	std::vector<std::vector<float>> layer1, layer2, output;

	layer1 = asLayerVector<float>(pt2, "l1");
	layer2 = asLayerVector<float>(pt2, "l2");
	output = asLayerVector<float>(pt2, "output");

	modelWeights.push_back(layer1);
	modelWeights.push_back(layer2);
	modelWeights.push_back(output);

	Net myNet(modelWeights);

	std::vector<double> inputVals, resultVals;
	// int trainingPass = 0;
	// while (trainingPass < 1)
	// {
	// 	++trainingPass;
	// 	std::cout << "Pass:" << trainingPass << "\n";

	// 	inputVals = as_vector<double>(pt2, "in");
	// 	showVectorVals<double>(": Inputs :", inputVals);
	// 	myNet.feedForward(inputVals);

	// 	// Collect the net's actual results:
	// 	myNet.getResults(resultVals);
	// 	showVectorVals<double>("Outputs:", resultVals);

	// 	// assert(targetVals.size() == topology.back());
	// }
	inputVals = as_vector<double>(pt2, "in");
	showVectorVals<double>(": Inputs :", inputVals);
	myNet.infer(inputVals, resultVals);
	showVectorVals<double>("Outputs:", resultVals);

	std::cout << std::endl
			  << "Done" << std::endl;
}