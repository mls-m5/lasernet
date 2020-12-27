#pragma once

#include "graph/inode.h"
#include "msl/range.h"

class FullLayer : public INode {
public:
    FullLayer(size_t inputSize, size_t outputSize)
        : _inputSize(inputSize), _parameterRowSize(inputSize + 1),
          _outputSize(outputSize) {}

    //! @see INode
    DataSize dataSize() const override {
        return {
            .input = _inputSize,
            .parameters = (_inputSize + 1) * _outputSize,
            .output = _outputSize,
        };
    }

    //! @see INode
    ConstSpanD input(ConstSpanD data) const override {
        return data;
    }

    //! @see INode
    ConstSpanD output(ConstSpanD data) const override {
        return data;
    }

    //! @see INode
    void calculateValues(CalculateArgs args) const override {
        auto &x = args.x;
        auto &parameters = args.parameters;
        auto &y = args.y;

        for (auto o : msl::range(_outputSize)) {
            y[o] = 0;
            // Calculate weights
            for (auto i : msl::range(_inputSize)) {
                y[o] += x[i] * cweights(parameters, i, o);
            }

            y[o] += cbias(parameters, o);
        }
    }

    //! @see INode
    void backpropagate(BackpropagateArgs args) const override {
        auto &dEdw = args.dEdw;
        auto &dEdx = args.dEdx;
        auto &dEdy = args.dEdy;
        auto &parameters = args.parameters;

        for (auto o : msl::range(_outputSize)) {
            for (auto i : msl::range(_inputSize)) {
                weights(dEdw, i, o) = dEdy[o] * args.x[i];
            }

            // Derivatives of the bias
            bias(dEdw, o) += dEdy[o];
        }

        for (auto i : msl::range(_inputSize)) {
            dEdx[i] = 0;
            for (auto o : msl::range(_outputSize)) {
                dEdx[i] += cweights(parameters, i, o) * dEdy[o];
            }
        }
    }

    //! Return a specific weight from the parameters from given index
    inline double &weights(SpanD parameters,
                           size_t input,
                           size_t output) const {
        return parameters[index(input, output)];
    }

    //! Const version
    inline double cweights(ConstSpanD parameters,
                           size_t input,
                           size_t output) const {
        return parameters[index(input, output)];
    }

    //! Return _const_ bias associated with a specific output
    //! @param output index of the output node
    inline double cbias(ConstSpanD parameters, size_t output) const {
        return parameters[index(_inputSize, output)];
    }

    //! Return bias associated with a specific output
    //! @param output index of the output nodeo
    inline double &bias(SpanD parameters, size_t output) const {
        return parameters[index(_inputSize, output)];
    }

private:
    inline size_t index(size_t input, size_t output) const {
        return input + output * _parameterRowSize;
    }

    size_t _inputSize;
    size_t _parameterRowSize;
    size_t _outputSize;
};
