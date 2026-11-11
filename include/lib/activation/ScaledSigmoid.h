#pragma once

#include <activation/ActivationFunction.h>

#include <string>

// Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
namespace dnn
{
    class ScaledSigmoid : ActivationFunction
    {
    public:
        ScaledSigmoid(double min, double max);

        double getMin() const;
        double getMax() const;

        double activation(double x) override;
        double activationBackwards(double x) override;
        std::string getName() const override;

    private:
        double min;
        double max;
    };

} // namespace dnn
