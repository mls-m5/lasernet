#pragma once

#include "dataset/dataset.h"
#include "optimizer/ioptimizer.h"
#include <vector>

class Trainer {

public:
    Trainer(IOptimizer &optimizer, Dataset dataset, size_t /*batchSize*/ = 1)
        : _dataset(std::move(dataset)), _optimizer(&optimizer)
    /*, _batchSize(batchSize) */ {}

    struct BackPropagationData {
        std::vector<double> dEdx; // Input derivative
        std::vector<double> dEdw; // Parameter derivative
        std::vector<double> output;

        BackPropagationData(INode::DataSize sizes)
            : dEdx(sizes.input), dEdw(sizes.parameters), output(sizes.output) {}
    };

    void step(/*IInput *input,*/ INode &node, ICostFunction &cost) {
        auto input = SpanD{_dataset.data.at(_currentDataset).x};
        auto expectedOutput = SpanD{_dataset.data.at(_currentDataset).y};

        auto sizes = node.dataSize();

        parameters.resize(sizes.parameters);

        // Todo: save between runs, one per thread and layer
        BackPropagationData data{sizes};

        // Output
        std::vector<double> outputDerivative(expectedOutput.size());

        node.calculateValues({
            .x = input,
            .parameters = parameters,
            .y = data.output,
        });

        cost.derive(node.output(data.output), expectedOutput, outputDerivative);

        node.backpropagate({
            .x = input,
            .parameters = parameters,
            .y = data.output,
            .dEdxPrev = outputDerivative,
            .dEdx = data.dEdx,
            .dEdw = data.dEdw,
        });

        constexpr double learningRate = .1;

        _optimizer->applyDerivative(data.dEdw, learningRate, parameters);

        if (++_currentDataset >= _dataset.data.size()) {
            _currentDataset = 0;
        }

        _lastCost = cost.cost(node.output(data.output), expectedOutput);
    }

    double cost() {
        return _lastCost;
    }

private:
    Dataset _dataset;
    std::vector<double> parameters; //! w
    IOptimizer *_optimizer;
    //    size_t _batchSize = 1;
    size_t _currentDataset = 0;
    double _lastCost = 0;
};
