#pragma once

#include "dataset/dataset.h"
#include "optimizer/ioptimizer.h"
#include <vector>

class Trainer {

public:
    Trainer(IOptimizer &optimizer, Dataset dataset, size_t /*batchSize*/ = 1)
        : _dataset(std::move(dataset)), _optimizer(&optimizer)
    /*, _batchSize(batchSize) */ {}

    void step(/*IInput *input,*/ INode &node, ICostFunction &cost) {
        auto input = SpanD{_dataset.data.at(_currentDataset).x};
        auto expectedOutput = SpanD{_dataset.data.at(_currentDataset).y};

        // Todo save these between runs, per thread
        //        std::vector<double> parameters(node.parameterSize());
        parameters.resize(node.parameterSize());
        std::vector<double> derivative(parameters.size());
        std::vector<double> activation(node.activationSize());

        // Output
        std::vector<double> outputDerivative(expectedOutput.size());

        node.calculateValues(input, parameters, activation);

        cost.derive(node.output(activation), expectedOutput, outputDerivative);

        node.backpropagate(
            input, parameters, activation, outputDerivative, derivative);

        constexpr double learningRate = .1;

        _optimizer->applyDerivative(derivative, learningRate, parameters);

        if (++_currentDataset >= _dataset.data.size()) {
            _currentDataset = 0;
        }

        _lastCost = cost.cost(node.output(activation), expectedOutput);
    }

    double cost() {
        return _lastCost;
    }

private:
    Dataset _dataset;
    std::vector<double> parameters;
    IOptimizer *_optimizer;
    //    size_t _batchSize = 1;
    size_t _currentDataset = 0;
    double _lastCost = 0;
};
