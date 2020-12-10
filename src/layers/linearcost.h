#pragma once

#include "graph/inode.h"

//! Layer that calculates the error beween train data and calculated output
class LinearCost : public ICostFunction {
public:
    // @see ICostFunction interface
    double cost(ConstSpanD activation, ConstSpanD expected) override {
        if (activation.size() != expected.size()) {
            throw std::runtime_error("error layer does not match data size");
        }

        auto iy = expected.begin();
        auto ia = activation.begin();

        double sum = 0;

        for (; ia != activation.end(); ++iy, ++ia) {
            auto val = *ia - *iy;
            sum = val * val;
        }

        return sum / 2.;
    }

    // @see ICostFunction interface
    void derive(ConstSpanD activation,
                ConstSpanD expected,
                SpanD derivative) override {
        if (activation.size() != expected.size()) {
            throw std::runtime_error("error layer does not match data size");
        }

        auto iy = expected.begin();
        auto ia = activation.begin();
        auto id = derivative.begin();

        for (; ia != activation.end(); ++iy, ++ia, ++id) {
            *id = *ia - *iy;
        }
    }
};
