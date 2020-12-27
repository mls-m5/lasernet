#pragma once

#include "dataset/dataset.h"
#include "fmt/core.h"
#include "msl/range.h"
#include "optimizer/ioptimizer.h"
#include <algorithm>
#include <vector>

class Trainer {

public:
    //! @param batchsize:
    //! 0: batch gradient descent (use the whole training set for each step)
    //! 1: gradient descent (use only one sample for each step)
    //! between 2 and ∞: mini-batch, (use the specified numbers of samples per
    //! step before updating)
    Trainer(INode &node,
            IOptimizer &optimizer,
            Dataset dataset,
            size_t batchSize = 1)
        : _dataset(std::move(dataset)), _optimizer(&optimizer),
          _batchSize(batchSize) {

        auto sizes = node.dataSize();
        _parameters.resize(sizes.parameters);

        for (auto &parameter : _parameters) {
            parameter = .1; // Make sure that they are not zero
        }

        _parameters.back() = 0; //! Todo: Remove this
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

    //! Step a single thread once
    //! That is calculate derivatives but do not apply them
    void step(const INode &node,
              const ICostFunction &cost,
              ConstSpanD input,
              ConstSpanD expectedOutput,
              BackPropagationData &data) {

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
            .dEdy = data.outputDerivative,
            .dEdx = data.dEdx,
            .dEdw = data.dEdw,
        });
    }

    void step(const INode &node, const ICostFunction &cost) {
        // Todo: save between runs, one per thread and layer

        auto sizes = node.dataSize();
        auto data = BackPropagationData{sizes};

        std::vector<double> dEdwSum(data.dEdw.size());
        double costSum = 0;

        const auto batchSize = _batchSize ? _batchSize : _dataset.data.size();

        for ([[maybe_unused]] auto i : msl::range(batchSize)) {

            auto input = ConstSpanD{_dataset.data.at(_currentDataset).x};
            auto expectedOutput =
                ConstSpanD{_dataset.data.at(_currentDataset).y};

            step(node, cost, input, expectedOutput, data);

            if (++_currentDataset >= _dataset.data.size()) {
                _currentDataset = 0;
                ++_epoch;
            }

            costSum += cost.cost(node.output(data.output), expectedOutput);

            for (auto i : msl::range(dEdwSum.size())) {
                dEdwSum[i] += data.dEdw[i];
            }
        }

        std::transform(dEdwSum.begin(),
                       dEdwSum.end(),
                       dEdwSum.begin(),
                       [div = static_cast<double>(batchSize)](double value) {
                           return value / div;
                       });

        _lastCost = costSum / batchSize;

        constexpr double learningRate = .01;

        fmt::print("   d: {}\t {}\n", dEdwSum.front(), dEdwSum.back());
        _optimizer->applyDerivative(dEdwSum, learningRate, _parameters);
    }

    double cost() const {
        return _lastCost;
    }

    //! Return the nmuber of epocs trained
    size_t epoch() const {
        return _epoch;
    }

    ConstSpanD parameters() const {
        return _parameters;
    }

private:
    Dataset _dataset;
    std::vector<double> _parameters; //! w
    IOptimizer *_optimizer;
    size_t _batchSize = 1;
    size_t _currentDataset = 0;
    double _lastCost = 0;
    size_t _epoch = 0;
};
