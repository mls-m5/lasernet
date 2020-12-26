#pragma once

#include "graph/inode.h"
#include <algorithm>

class Relu : public INode {
    size_t _inputSize;
    size_t _outputSize;

public:
    Relu(size_t inputSize, size_t outputSize)
        : _inputSize(inputSize), _outputSize(outputSize) {}

    //! @see INode
    DataSize dataSize() override {
        return {
            .input = _inputSize,
            .parameters = 0,
            .output = _outputSize,
        };
    }

    //! @see INode
    ConstSpanD input(ConstSpanD data) override {
        return data;
    }

    //! @see INode
    ConstSpanD output(ConstSpanD data) override {
        return data;
    }

    //! @see INode
    void calculateValues(CalculateArgs args) override {
        if (args.x.size() != args.y.size()) {
            throw std::range_error{"input and output does not match in " +
                                   std::string{__FILE__}};
        }

        std::transform(args.x.begin(),
                       args.x.end(),
                       args.y.begin(),
                       [](double val) { return std::max(val, 0.); });
    }

    //! @see INode
    void backpropagate(BackpropagateArgs) override {
        throw std::runtime_error("not implemented");
    }
};
