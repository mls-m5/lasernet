#pragma once

#include "graph/inode.h"

//! Offset one input to one output
//! This class is intended for testing
class SingleBias : public INode {
public:
    //    //! @see INode
    //    size_t parameterSize() {
    //        return 1;
    //    }

    //    //! @see INode
    //    size_t activationSize() {
    //        return 1;
    //    }

    //! @see INode
    DataSize dataSize() override {
        return {
            .input = 1,
            .parameters = 1,
            .output = 1,
        };
    }

    //! @see INode
    ConstSpanD output(ConstSpanD data) override {
        return data;
    }

    //! @see INode
    void calculateValues(CalculateArgs args) override {
        args.y.front() = args.x.front() + args.parameters.front();
    }

    //! @see INode
    void backpropagate(BackpropagateArgs args) override {
        args.dEdw.front() = 1;
        args.dEdx.front() = 1;
    }
};
