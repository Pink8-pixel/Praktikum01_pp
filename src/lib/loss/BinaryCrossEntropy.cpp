#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn
{
    double BinaryCrossEntropy::getCostValue(const std::vector<double> &predictedOutput, const std::vector<double> &actualOutput) const
    {
        auto n = predictedOutput.size();
        std::vector<double> P;
        std::vector<double> Q;
        std::vector<double> R;

        P.reserve(n);
        Q.reserve(n);
        R.reserve(n);

        for (size_t i = 0; i < n; i++)
        {
            const double p = clampProbability(predictedOutput[i]);

            P.push_back(std::log(p));
            Q.push_back(std::log(1.0 - p));
            R.push_back(1.0 - actualOutput[i]);
        }

        double cost = -(dotProduct(actualOutput, P) + dotProduct(R, Q));

        return cost / static_cast<double>(n);
    }

    double BinaryCrossEntropy::getAccuracyValue(const std::vector<double> &predictedOutput, const std::vector<double> &actualOutput) const
    {
        std::vector<double> V;
        V.reserve(predictedOutput.size());

        // this is dumb, i think actualOutput.size = predictedOutput.size but he said nothing about them being same size just the number of similarities
        for (size_t i = 0; i < predictedOutput.size(); i++)
        {
            V.push_back((predictedOutput[i] > 0.5) ? 1.0 : 0);
        }
        const std::size_t count = std::min(V.size(), actualOutput.size());

        size_t correct = 0;
        for (size_t i = 0; i < count; i++)
        {
            if (V[i] == actualOutput[i])
            {
                ++correct;
            }
        }

        /*
            i'd do it like this
            size_t correct = 0;
            for (size_t i = 0; i < predictedOutput.size(); i++)
            {
                double predicted_conv = predictedOutput[i] > 0.5 ? 1.0 : 0.0;
                if (predicted_conv == actualOutput[i])
                {
                    ++correct;
                }

            }
        */

        return static_cast<double>(correct) / static_cast<double>(predictedOutput.size());
    }

    std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double> &predictedOutput, const std::vector<double> &actualOutput) const
    {
        const double n = static_cast<double>(predictedOutput.size());
        std::vector<double> gradient(predictedOutput.size());
        
        for (std::size_t i = 0; i < predictedOutput.size(); ++i)
        {
            const double p = clampProbability(predictedOutput[i]);
            const double y = actualOutput[i];

            gradient[i] = -((y / p) - ((1.0 - y) / (1.0 - p))) /n;
        }
        return gradient;
    }

    std::string BinaryCrossEntropy::getName() const
    {
        return "binary_cross_entropy";
    }

}
