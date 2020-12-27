#pragma once

#include "graph/inode.h"

//! Kind iof the simplest possible model
//! One possible input. One "wheight" = derivative and one bias = constant
//! This is supposed to be used mostly for testing
//! Notice that this is a special case of a fully connected layer
class SingleStraightLine : public INode {
    //! Variables reqired to calculate a straight line (in this way)
    static constexpr size_t kIndex = 0;
    static constexpr size_t mIndex = 1;

public:
    //! @see INode
    DataSize dataSize() const override {
        return {
            .input = 1,
            .parameters = 2,
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
        args.y[0] =
            args.parameters[kIndex] * args.x.front() + args.parameters[mIndex];
    }

    //! @see INode
    void backpropagate(BackpropagateArgs args) const override {
        args.dEdw.front() = args.parameters[kIndex] * args.dEdxPrev.front();
        args.dEdw.back() = args.dEdxPrev.front();

        //        args.dEdx.front() = args.parameters[kIndex] *
        //        args.dEdxPrev.front();
        args.dEdx.front() =
            args.dEdw.front(); // Its thes same for a straight line
    }
};
