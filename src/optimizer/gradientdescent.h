#pragma once

#include "optimizer/ioptimizer.h"
#include <vector>

class GradientDescent : public IOptimizer {
public:
    void applyDerivative(ConstSpanD derivative,
                         double learningRate,
                         SpanD parameters) {
        if (parameters.size() != derivative.size()) {
            throw std::invalid_argument(
                "parameter and derivative size does not match");
        }
        else if (learningRate <= 0) {
            throw std::invalid_argument("learning rate cannot be below 0");
        }
        else if (learningRate > 1) {
            throw std::invalid_argument("learning rate cannot be above 1");
        }

        auto parameterIt = parameters.begin();
        auto derivativeIt = derivative.begin();
        for (; parameterIt < parameters.end(); ++parameterIt, ++derivativeIt) {
            *parameterIt -= *derivativeIt * learningRate;
        }
    }
};
