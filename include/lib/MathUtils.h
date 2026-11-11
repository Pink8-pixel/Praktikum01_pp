#pragma once

#include <vector>
#include <algorithm>
namespace dnn
{
    template <typename T>
    std::vector<double> normalize(const std::vector<T> &values, int maxValue)
    {
        assert(!values.empty());
        std::vector<double> r;
        r.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++)
        {
            r.push_back(static_cast<double>(values[i]) / maxValue);
        }
        return r;
    }

    template <typename T>
    std::vector<double> normalize(const std::vector<T> &values)
    {
        assert(!values.empty());
        const T *maxValue = std::max_element(values.begin(), values.end());
        return normalize(values, static_cast<int>(*maxValue));
    }

    double dotProduct(const std::vector<double> &a, const std::vector<double> &b);
    double sum(const std::vector<double> &values);

    std::vector<double> scaleVector(const std::vector<double> &values, const double factor);
    std::vector<double> addVectors(const std::vector<double> &a, const std::vector<double> &b);
    
    
    std::vector<std::vector<double>> scaleMat(const std::vector<std::vector<double>> &m, const double factor);
    std::vector<std::vector<double>> addMats(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b);

    // Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
    // and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
    std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);

}
