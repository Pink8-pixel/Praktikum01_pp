#include <activation/Sigmoid.h>
#include <cmath>
namespace dnn
{
    double Sigmoid::activation(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }
    double Sigmoid::activationBackwards(double x)
    {
        return activation(x) * (1 - activation(x));
    }
    std::string Sigmoid::getName() const
    {
        return "sigmoid";
    }

} // namespace dnn
