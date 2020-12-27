#pragma once

#include "graph/inode.h"

//! Offset one input to one output
//! This class is intended for testing
class SingleBias : public INode {
public:
    //! @see INode
    DataSize dataSize() const override {
        return {
            .input = 1,
            .parameters = 1,
            .output = 1,
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
        args.y.front() = args.x.front() + args.parameters.front();
    }

    //! @see INode
    void backpropagate(BackpropagateArgs args) const override {
        args.dEdw.front() = args.dEdxPrev.front();
        args.dEdx.front() = args.dEdxPrev.front();
    }
};
