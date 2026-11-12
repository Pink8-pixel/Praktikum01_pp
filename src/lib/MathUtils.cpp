#include <MathUtils.h>

#include <stdexcept>
#include <string>

namespace dnn
{
    double dotProduct(const std::vector<double> &a, const std::vector<double> &b)
    {
        double result = 0;
        for (size_t i = 0; i < a.size(); i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    double sum(const std::vector<double> &values)
    {
        double s = 0;
        for (size_t i = 0; i < values.size(); i++)
        {
            s += values[i];
        }
        return s;
    }

    std::vector<double> scaleVector(const std::vector<double> &values, double factor)
    {
        std::vector<double> result = values;
        for (double &value : result)
        {
            value *= factor;
        }
        return result;
    }

    std::vector<double> addVectors(const std::vector<double> &a, const std::vector<double> &b)
    {
        std::vector<double> result = a;
        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] += b[i];
        }
        return result;
    }

    std::vector<std::vector<double>> scaleMat(const std::vector<std::vector<double>> &m, const double factor)
    {
        std::vector<std::vector<double>> result = m;
        for (std::vector<double> &row : result)
        {
            for (double &value : row)
            {
                value *= factor;
            }
        }
        return result;
    }

    std::vector<std::vector<double>> addMats(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
    {
        auto result = a;
        for (size_t i = 0; i < result.size(); ++i)
        {

            for (size_t j = 0; j < result[i].size(); ++j)
            {
                result[i][j] += b[i][j];
            }
        }
        return result;
    }

    std::vector<double> encodeAsOneHot(double value, std::size_t numClasses)
    {
        if (value < 0.0 || value >= static_cast<double>(numClasses))
        {
            throw std::out_of_range("encodeAsOneHot: value " + std::to_string(value) + " is out of range for " +
                                    std::to_string(numClasses) + " classes");
        }

        std::vector<double> encoded(numClasses, 0.0);
        encoded[static_cast<std::size_t>(value)] = 1.0;
        return encoded;
    }

}
