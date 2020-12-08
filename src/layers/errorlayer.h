#pragma once

#include "network/inode.h"

//! Layer that calculates the error beween train data and calculated output
class LinearCost : public ICostFunction {
public:
    ConstSpanD _y;

    void setData(ConstSpanD y) override {
        _y = y;
    }

    size_t size() override {
        return _y.size();
    }

    //! @param activation is the activation of the last layer before this
    void derive(ConstSpanD activation, SpanD derivative) override {
        if (activation.size() != _y.size()) {
            throw std::runtime_error("error layer does not match data size");
        }

        auto iy = _y.begin();
        auto ia = activation.begin();
        auto id = derivative.begin();

        for (; ia != activation.end(); ++ia, ++ia, ++id) {
            *id = *iy - *ia;
        }
    }
};
