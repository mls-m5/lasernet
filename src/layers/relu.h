#pragma once

#include "graph/inode.h"
#include "msl/range.h"
#include <algorithm>

class Relu : public INode {
    size_t _size;

public:
    Relu(size_t size) : _size(size) {}

    //! @see INode
    DataSize dataSize() const override {
        return {
            .input = _size,
            .parameters = 0,
            .output = _size,
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
    void backpropagate(BackpropagateArgs args) const override {
        for (auto i : msl::range(args.y.size())) {
            args.dEdx[i] = args.y[i] ? args.dEdy[i] : 0;
        }
    }
};
