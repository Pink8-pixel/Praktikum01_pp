#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn
{
    Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction *>> layerParameters, LossFunction *lossFunction)
    {
        setLossFunction(lossFunction);

        for (size_t i = 0; i < layerParameters.size(); i++) // number of layers=layerParameters.size()
        {
            auto &[weights, bias, activationFunction] = layerParameters[i]; // extract parameters

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

        if (layerId == 0)
        {
            return initialInput;
        }

        return layers[layerId - 1]->getActivationOutput();
    }

    void Network::fullForwardPropagation(const std::vector<double> &input)
    {
        initialInput = input;

        if (layers.empty())
        {
            output = input;
            return;
        }

        for (size_t i = 0; i < layers.size(); i++)
        {
            layers[i]->forwardPropagation(getLayerInput(i));
        }
        output = layers.back()->getActivationOutput();
    }

    void Network::fullBackwardPropagation(const std::vector<double> &predictedOutput, const std::vector<double> &actualOutput)
    {
        if (ownedLossFunction == nullptr)
        {
            throw std::logic_error("Cannot backpropagate: no loss function has been set.");
        }

        std::vector<double> dACurr = ownedLossFunction->getGradient(predictedOutput, actualOutput);

        for (size_t i = layers.size(); i-- > 0;)
        {
            layers[i]->backwardPropagation(dACurr, getLayerInput(i));
            dACurr = layers[i]->getDPreviousLayerOutput();
        }
    }

    
    void Network::trainBatch(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &actualOutputs, std::size_t batchSize,
                             int epochs, double learningRate)
    {

        // all the checks i wanna die
        if (ownedLossFunction == nullptr)
        {
            throw std::logic_error("Cannot train: no loss function has been set.");
        }
        if (layers.empty())
        {
            throw std::logic_error("Cannot train: the network has no layers.");
        }
        if (inputs.empty() || actualOutputs.empty())
        {
            throw std::invalid_argument("Training inputs and outputs must not be empty.");
        }
        if (inputs.size() != actualOutputs.size())
        {
            throw std::invalid_argument("Training inputs and outputs must have the same number of samples.");
        }
        if (batchSize <= 0)
        {
            throw std::invalid_argument("Batch size must be greater than zero.");
        }
        if (epochs < 0)
        {
            throw std::invalid_argument("Epoch count must not be negative.");
        }

        costHistory.clear();
        accuracyHistory.clear();

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double epochCost = 0.0;
            double epochAccuracy = 0.0;

            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                fullForwardPropagation(inputs[i]);

                epochCost += ownedLossFunction->getCostValue(output, actualOutputs[i]);
                epochAccuracy += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);

                fullBackwardPropagation(output, actualOutputs[i]);

                const bool endOfBatch = ((i + 1) % batchSize == 0);
                const bool endOfDataset = (i + 1 == inputs.size());

                if (endOfBatch || endOfDataset)
                {
                    for (Layer *layer : layers)
                    {
                        layer->commitToWeights(learningRate);
                    }
                }
            }

            costHistory.push_back(epochCost / static_cast<double>(inputs.size()));
            accuracyHistory.push_back(epochAccuracy / static_cast<double>(inputs.size()));
        }
    }

    double Network::computeAccuracy(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &actualOutputs)
    {
        if (ownedLossFunction == nullptr)
        {
            throw std::logic_error("Cannot compute accuracy: no loss function has been set.");
        }
        if (layers.empty())
        {
            throw std::logic_error("Cannot compute accuracy: the network has no layers.");
        }
        if (inputs.empty() || actualOutputs.empty())
        {
            throw std::invalid_argument("Inputs and outputs must not be empty.");
        }
        if (inputs.size() != actualOutputs.size())
        {
            throw std::invalid_argument("Inputs and outputs must have the same number of samples.");
        }

        double totalCorrect = 0.0;
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            fullForwardPropagation(inputs[i]);
            totalCorrect += ownedLossFunction->getAccuracyValue(output, actualOutputs[i]);
        }

        return totalCorrect / static_cast<double>(inputs.size());
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
