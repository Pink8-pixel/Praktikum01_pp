#include <Layer.h>

#include <MathUtils.h>

#include <cmath>
#include <iostream>
#include <utility>

namespace dnn
{
    Layer::Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
                 ActivationFunction *activationFunction, std::string name)
        : id(id), name(std::move(name)), seed(seed), layerInputSize(layerInputSize), layerOutputSize(layerOutputSize),
          activationFunction(activationFunction)
    {
        accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
        accumulatedDBias.assign(layerOutputSize, 0.0);
    }

    Layer::~Layer()
    {
        delete activationFunction;
    }

    void Layer::init()
    {
        std::mt19937 randomGenerator(seed);

        // Glorot/Xavier uniform initialization:
        // scales the weight range by fan-in/fan-out so pre-activation variance stays roughly O(1)
        // regardless of layer width, instead of growing with layerInputSize the way a fixed
        // [-1, 1] range does.
        const double limit = std::sqrt(6.0 / static_cast<double>(layerInputSize + layerOutputSize));
        std::uniform_real_distribution<double> distribution(-limit, limit);

        weights.assign(layerOutputSize, std::vector<double>(layerInputSize));
        for (auto &row : weights)
        {
            for (double &value : row)
            {
                value = distribution(randomGenerator);
            }
        }

        // Zero-initialized bias
        bias.assign(layerOutputSize, 0.0);
    }

    void Layer::forwardPropagation(const std::vector<double> &previousLayerOutput)
    {
        preActivationOutput.assign(layerOutputSize, 0.0);
        activationOutput.assign(layerOutputSize, 0.0);

        for (size_t i = 0; i < layerOutputSize; i++)
        {
            preActivationOutput[i] = dotProduct(previousLayerOutput, weights[i]) + bias[i];
            activationOutput[i] = activationFunction->activation(preActivationOutput[i]);
        }
    }

    void Layer::backwardPropagation(const std::vector<double> &dActivationOutput, const std::vector<double> &previousLayerOutput)
    {
        std::vector<double> dPreActivation(layerOutputSize, 0.0);
        for (size_t i = 0; i < layerOutputSize; i++)
        {
            dPreActivation[i] = dActivationOutput[i] * activationFunction->activationBackwards(preActivationOutput[i]);
        }
        dWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));

        for (size_t i = 0; i < accumulatedDWeights.size(); ++i)
        {
            accumulatedDWeights[i] = addVectors(accumulatedDWeights[i], dWeights[i]);
        }

        dBias = dPreActivation;

        accumulatedDBias = addVectors(accumulatedDBias, dBias);

        numProcessedSamples += 1;

        dPreviousLayerOutput.assign(layerInputSize, 0.0);

        for (size_t i = 0; i < layerInputSize; i++)
        {
            dPreviousLayerOutput[i] = dotProduct(getWeightsColumn(i), dPreActivation);
        }
    }

    std::vector<double> Layer::getWeightsColumn(int index)
    {
        std::vector<double> result(weights.size());
        for (size_t i = 0; i < weights.size(); i++)
        {
            result[i] = weights[i][index];
        }
        return result;
    }

    void Layer::updateParameters(const std::vector<std::vector<double>> &dWeights, const std::vector<double> &dBias, double learningRate){
        auto scaled_dWeights = scaleMat(dWeights, -learningRate);
        auto updatedWeights = addMats(weights,scaled_dWeights);
        weights = updatedWeights;
        auto scaled_dBias = scaleVector(dBias, -learningRate);
        auto updatedBias = addVectors(bias, scaled_dBias);
        bias = updatedBias;
    }

    void Layer::commitToWeights(double learningRate){
        auto scaled_accumulatedDWeight = scaleMat(accumulatedDWeights,1.0/numProcessedSamples);
        auto scaled_accumulatedDBias = scaleVector(accumulatedDBias,1.0/numProcessedSamples);
        
        accumulatedDWeights = scaled_accumulatedDWeight;
        accumulatedDBias = scaled_accumulatedDBias;

        updateParameters(accumulatedDWeights,accumulatedDBias,learningRate);
        accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
        accumulatedDBias.assign(layerOutputSize, 0.0);
        numProcessedSamples = 0;

    }


    void Layer::setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias)
    {
        this->weights = std::move(weights);
        this->bias = std::move(bias);
    }

    uint64_t Layer::getId() const
    {
        return id;
    }

    const std::string &Layer::getName() const
    {
        return name;
    }

    const std::vector<double> &Layer::getActivationOutput() const
    {
        return activationOutput;
    }

    const std::vector<double> &Layer::getPreActivationOutput() const
    {
        return preActivationOutput;
    }

    ActivationFunction *Layer::getActivationFunction() const
    {
        return activationFunction;
    }

    const std::vector<std::vector<double>> &Layer::getWeights() const
    {
        return weights;
    }

    const std::vector<double> &Layer::getBias() const
    {
        return bias;
    }

    const std::vector<double> &Layer::getDPreviousLayerOutput() const
    {
        return dPreviousLayerOutput;
    }

    const std::vector<std::vector<double>> &Layer::getDWeights() const
    {
        return dWeights;
    }

    const std::vector<double> &Layer::getDBias() const
    {
        return dBias;
    }

}
