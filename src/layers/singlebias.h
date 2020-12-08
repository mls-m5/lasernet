#pragma once

#include "network/inode.h"

//! Offset one input to one output
//! This class is intended for testing
class SingleBias : public INode {
public:
    //! @see INode
    size_t parameterSize() {
        return 1;
    }

    //! @see INode
    size_t activationSize() {
        return 1;
    }

    //! @see INode
    ConstSpanD output(ConstSpanD data) {
        return data;
    }

    //! @see INode
    void calculateValues(ConstSpanD x, ConstSpanD parameters, SpanD y) {
        y[0] = x[0] + parameters[0];
    }

    //! @see INode
    void backpropagate(ConstSpanD /*x*/,
                       ConstSpanD /*parameters*/,
                       ConstSpanD /*y*/,
                       ConstSpanD previousDerivative,
                       SpanD derivative) {
        derivative[0] = previousDerivative[0];
    }
};
