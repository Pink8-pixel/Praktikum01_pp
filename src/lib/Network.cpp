#include <Network.h>

#include <iostream>
#include <stdexcept>
#include <assert.h>

namespace dnn
{
    Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction *>> layerParameters, LossFunction *lossFunction)
    {
        setLossFunction(lossFunction);

        for (size_t i = 0; i < layerParameters.size(); i++) // number of layers=layerParameters.size()
        {
            auto &[weights, bias, activationFunction] = layerParameters[i]; // extract parameters
            assert(!weights.empty());

            const auto inputSize = weights[0].size();
            const auto outputSize = weights.size();

            Layer *layer = new Layer(i, nextSeed++, inputSize, outputSize, activationFunction); // construct the create a newLayer with the set size
            layer->setParameters(weights, bias);                                                // assign the parameters to the layer
            layers.push_back(layer);
        }
    }

    Network::~Network()
    {
        delete this->ownedLossFunction;
        for (Layer *l : layers)
        {
            delete l;
        }
    }

    Network &Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction *activationFunction)
    {
        Layer *layer = new Layer(layers.size(), nextSeed++, inputDim, outputDim, activationFunction);
        layer->init();
        layers.push_back(layer);
        return *this;
    }

    const std::vector<double> &Network::getLayerInput(uint64_t layerId) const
    {
        assert(layerId <= layers.size());

        if (layerId == 0)
        {
            return initialInput;
        }

        return layers[layerId - 1]->getActivationOutput();
    }

    void Network::setLossFunction(LossFunction *lossFunction)
    {
        delete this->ownedLossFunction;
        this->ownedLossFunction = lossFunction;
    }

    const LossFunction *Network::getLossFunction() const
    {
        return ownedLossFunction;
    }

    const std::vector<double> &Network::getOutput() const
    {
        return output;
    }

    const std::vector<double> &Network::getCostHistory() const
    {
        return costHistory;
    }

    const std::vector<double> &Network::getAccuracyHistory() const
    {
        return accuracyHistory;
    }

    const std::vector<Layer *> &Network::getLayers() const
    {
        return layers;
    }

    void Network::setLayers(std::vector<Layer *> newLayers)
    {
        for (Layer *layer : layers)
        {
            delete layer;
        }
        layers = std::move(newLayers);
        initialInput.clear();
        output.clear();
        costHistory.clear();
        accuracyHistory.clear();
    }

}
