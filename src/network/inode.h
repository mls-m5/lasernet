#pragma once

#include "util/span.h"

class INode {
    //! First step to get derivatives
    //! @param parameters is the values for weight etc
    //! @param activation is the output from the function, this will in turn be
    //! passed on to the backpropagation function
    virtual void calculateValues(const Span<double> parameters,
                                 SpanD activation) = 0;

    //! Calculate the derivative of a node given parameters and output of
    //! previous calculation
    //! @param derivative output result from node
    virtual void backpropagate(const SpanD parameters,
                               const SpanD activation,
                               const SpanD previousDerivative,
                               SpanD derivative) = 0;

    //! Number of values used for calculating
    virtual size_t memorySize() = 0;
};
