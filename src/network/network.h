#pragma once

#include "inode.h"

//! A network is a collection of nodes that can be used as a single node
class Network : INode {
    Network() {}

    // @see INode
    size_t parameterSize() override {
        // Todo: Implement
    }

    // @see INode
    size_t activationSize() override {
        // Todo: Implement
    }

    ConstSpanD output(ConstSpanD data) override {
        // Todo: Implement
    }

    // @see INode
    void calculateValues(ConstSpanD x,
                         ConstSpanD parameters,
                         SpanD y) override {
        // Todo: Implement
    }

    // @see INode
    void backpropagate(ConstSpanD x,
                       ConstSpanD parameters,
                       ConstSpanD activation,
                       ConstSpanD previousDerivative,
                       SpanD derivative) override {
        // Todo: Implumentt
    }
};
