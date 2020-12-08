#pragma once

#include "optimizer/ioptimizer.h"
#include <vector>

class Trainer {

public:
    Trainer(IOptimizer &optimizer) : _optimizer(&optimizer) {}

    void step(IInput *input, INode *node, ICostFunction *cost) {
        // Thread safety if doing multi threaded?
        // Might needs to be copied
        auto inputActivation = input->input();

        // Todo save these between runs, per thread
        std::vector<double> parameters(node->parameterSize());
        std::vector<double> derivative(parameters.size());
        std::vector<double> activation(node->activationSize());

        // Output
        std::vector<double> outputDerivative(cost->size());

        node->calculateValues(inputActivation, parameters, activation);

        cost->derive(node->output(activation), outputDerivative);

        node->backpropagate(inputActivation,
                            parameters,
                            activation,
                            outputDerivative,
                            derivative);

        constexpr double learningRate = .1;

        _optimizer->applyDerivative(derivative, learningRate, parameters);
    }

private:
    IOptimizer *_optimizer;
};
