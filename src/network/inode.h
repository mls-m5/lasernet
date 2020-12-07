#pragma once

#include "util/span.h"
#include <cstdint>

class INode {

    //! Number of values used for calculating
    virtual size_t parameterSize() = 0;

    //! Size of values to keep temporary data
    virtual size_t activationSize() = 0;

    //! First step to get derivatives
    //! @param x is the attributes/features/input to calculate on
    //! @param parameters is the values for weights, biases, etc
    //! @param activation/y is the output from the function, this will in turn
    //! be passed on to the backpropagation function
    virtual void calculateValues(const SpanD x,
                                 const Span<double> parameters,
                                 SpanD y) = 0;

    //! Calculate the derivative of a node given parameters and output of
    //! previous calculation
    //! @param derivative output result from node
    virtual void backpropagate(const SpanD x,
                               const SpanD parameters,
                               const SpanD y,
                               const SpanD previousDerivative,
                               SpanD derivative) = 0;
};
