#pragma once

#include "optimizer/ioptimizer.h"
#include <vector>

class GradientDescent : public IOptimizer {
public:
    void applyDerivative(ConstSpanD derivative,
                         double learningRate,
                         SpanD parameters) {
        if (parameters.size() != derivative.size()) {
            throw std::runtime_error(
                "parameter and derivative size does not match");
        }
        auto ip = parameters.begin();
        auto id = derivative.begin();
        for (; ip < parameters.end(); ++ip, ++id) {
            *ip -= *id * learningRate;
        }
    }
};
