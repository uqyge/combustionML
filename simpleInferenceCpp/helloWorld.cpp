#include <iostream>
#include "nnInferModel.H"
using boost::property_tree::ptree;

int main()
{
    std::vector<std::vector<std::vector<float>>> modelWeights;
    ptree pt;
    modelWeights = loadModelWeights("data.json", pt);
    // for(auto &i:modelWeights){
    //     std::cout<<i.size()<<'\n';
    // }

    Net sample(modelWeights);

    std::vector<double> inputVals;

    // inputVals = as_vector<double>(pt, "in");
    inputVals = {0.042, 0.722, 0.22, 0.042, 0.722, 0.22, 0.3, 0.948, 0.33};
    // showVectorVals<double>(": Inputs :", inputVals);

    for (int i = 0; i < 3; ++i)
    {
        std::vector<double> resultVals;
        sample.infer(inputVals, resultVals);
        showVectorVals<double>("Outputs:", resultVals);
    }

    std::cout << std::endl
              << "Done" << std::endl;

    return 0;
}