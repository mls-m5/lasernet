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

        auto sizes = node.dataSize();

        parameters.resize(sizes.parameters);
        std::vector<double> dEdw(parameters.size());
        std::vector<double> dEdx(sizes.input);
        std::vector<double> activation(sizes.output);

        // Output
        std::vector<double> outputDerivative(expectedOutput.size());

        node.calculateValues({
            .x = input,
            .parameters = parameters,
            .y = activation,
        });

        cost.derive(node.output(activation), expectedOutput, outputDerivative);

        node.backpropagate({
            .x = input,
            .parameters = parameters,
            .y = activation,
            .dEdxPrev = outputDerivative,
            .dEdx = dEdx,
            .dEdw = dEdw,
        });

        constexpr double learningRate = .1;

        _optimizer->applyDerivative(dEdw, learningRate, parameters);

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
