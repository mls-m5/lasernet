#pragma once

#include "network/inode.h"

//! Kind iof the simplest possible model
//! One possible input. One "wheight" = derivative and one bias = constant
class SingleStraightLine : INode {
    //! Variables reqired to calculate a straight line (in this way)
    static constexpr size_t kIndex = 0;
    static constexpr size_t mIndex = 1;

    // @see INode
    size_t parameterSize() override {
        return 2;
    }

    // @see INode
    size_t activationSize() override {
        return 1; // A lines derivative only needs
    }

    // @see INode
    void calculateValues(const SpanD x,
                         const SpanD parameters,
                         SpanD y) override {
        y.at(0) = parameters.at(kIndex) * x.front() + parameters.at(mIndex);
    }

    // @see INode
    void backpropagate(const SpanD x,
                       const SpanD parameters,
                       const SpanD activation,
                       const SpanD previousDerivative,
                       SpanD derivative) override {
        derivative.front() = parameters.at(kIndex) * previousDerivative.front();
    }
};
