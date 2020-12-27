#pragma once

#include "dataset/dataset.h"
#include "optimizer/ioptimizer.h"
#include <vector>

class Trainer {

public:
    Trainer(INode &node,
            IOptimizer &optimizer,
            Dataset dataset,
            size_t /*batchSize*/ = 1)
        : _dataset(std::move(dataset)), _optimizer(&optimizer)
    /*, _batchSize(batchSize) */ {

        auto sizes = node.dataSize();
        _parameters.resize(sizes.parameters);

        for (auto &parameter : _parameters) {
            parameter = .1; // Make sure that they are not zero
        }
    }

    struct BackPropagationData {
        std::vector<double> dEdx; // Input derivative
        std::vector<double> dEdw; // Parameter derivative
        std::vector<double> output;
        std::vector<double> outputDerivative;

        BackPropagationData(INode::DataSize sizes)
            : dEdx(sizes.input), dEdw(sizes.parameters), output(sizes.output),
              outputDerivative(sizes.output) {}
    };

    void step(const INode &node, const ICostFunction &cost) {
        auto input = SpanD{_dataset.data.at(_currentDataset).x};
        auto expectedOutput = SpanD{_dataset.data.at(_currentDataset).y};

        auto sizes = node.dataSize();

        // Todo: save between runs, one per thread and layer
        BackPropagationData data{sizes};

        node.calculateValues({
            .x = input,
            .parameters = _parameters,
            .y = data.output,
        });

        cost.derive(
            node.output(data.output), expectedOutput, data.outputDerivative);

        node.backpropagate({
            .x = input,
            .parameters = _parameters,
            .y = data.output,
            .dEdxPrev = data.outputDerivative,
            .dEdx = data.dEdx,
            .dEdw = data.dEdw,
        });

        constexpr double learningRate = .1;

        _optimizer->applyDerivative(data.dEdw, learningRate, _parameters);

        if (++_currentDataset >= _dataset.data.size()) {
            _currentDataset = 0;
        }

        _lastCost = cost.cost(node.output(data.output), expectedOutput);
    }

    double cost() const {
        return _lastCost;
    }

    ConstSpanD parameters() const {
        return _parameters;
    }

private:
    Dataset _dataset;
    std::vector<double> _parameters; //! w
    IOptimizer *_optimizer;
    //    size_t _batchSize = 1;
    size_t _currentDataset = 0;
    double _lastCost = 0;
};
