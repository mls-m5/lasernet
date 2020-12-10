#pragma once

#include "graph/inode.h"

//! Kind iof the simplest possible model
//! One possible input. One "wheight" = derivative and one bias = constant
//! This is supposed to be used mostly for testing
class SingleStraightLine : public INode {
    //! Variables reqired to calculate a straight line (in this way)
    static constexpr size_t kIndex = 0;
    static constexpr size_t mIndex = 1;

public:
    // @see INode
    size_t parameterSize() override {
        return 2;
    }

    // @see INode
    size_t activationSize() override {
        return 1; // A lines only needs one output value
    }

    ConstSpanD output(ConstSpanD data) override {
        return data;
    }

    // @see INode
    void calculateValues(ConstSpanD x,
                         ConstSpanD parameters,
                         SpanD y) override {
        y[0] = parameters[kIndex] * x.front() + parameters[mIndex];
    }

    // @see INode
    void backpropagate(ConstSpanD /*x*/,
                       ConstSpanD /*parameters*/,
                       ConstSpanD /*activation*/,
                       ConstSpanD previousDerivative,
                       SpanD derivative) override {
        derivative.front() =
            /*parameters[kIndex] **/ previousDerivative.front();
        derivative.back() = 1;
    }
};
