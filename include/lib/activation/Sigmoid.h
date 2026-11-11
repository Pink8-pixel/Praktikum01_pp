#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn
{
    class Sigmoid : ActivationFunction
    {
    public:
        Sigmoid() = default;
        ~Sigmoid() = default;

        double activation(double x) override;
        double activationBackwards(double x) override;
        std::string getName() const override;
    };
} // namespace dnn
