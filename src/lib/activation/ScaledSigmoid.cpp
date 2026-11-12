#include <activation/ScaledSigmoid.h>
#include <cmath>
namespace dnn
{

    ScaledSigmoid::ScaledSigmoid(double min, double max) : min(min), max(max)
    {
    }
    double ScaledSigmoid::activation(double x)
    {
        const double y = (max - min) * (1.0 / (1.0 + std::exp(-x)));
        return min + y;
    }
    double ScaledSigmoid::activationBackwards(double x)
    {
        const double y = (1.0 / (1.0 + std::exp(-x)));
        const double z = max - min;

        return z * y * (1 - y);
    }
    std::string ScaledSigmoid::getName() const
    {
        return "scaled_sigmoid";
    }

    double ScaledSigmoid::getMin() const
    {
        return min;
    }
    double ScaledSigmoid::getMax() const
    {
        return max;
    }

} // namespace dnn
