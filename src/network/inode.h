#pragma once

#include <cstdint>
#include <gsl/span>

using SpanD = gsl::span<double>;
using ConstSpanD = gsl::span<const double>;

class ICostFunction {
public:
    virtual ~ICostFunction() = default;

    //    virtual void setData(ConstSpanD y) = 0;

    //! Calculate the "derivative", ie the error to backpropagate
    //! @param activation is the activation of the last layer before this
    virtual void derive(ConstSpanD activation,
                        ConstSpanD expected,
                        SpanD derivative) = 0;

    //    virtual size_t size() = 0;
};

// class IInput {
// public:
//    virtual ~IInput() = default;

//    virtual ConstSpanD input() = 0;
//};

class INode {
public:
    virtual ~INode() = default;

    //! Number of values used for calculating
    virtual size_t parameterSize() = 0;

    //! Size of values to keep temporary data
    virtual size_t activationSize() = 0;

    //! How much of the data is being sent to the next layer
    //! Returns the portion of the activation data that is going to be used
    //! If all is to be used, just return data
    virtual ConstSpanD output(ConstSpanD data) = 0;

    //! First step to get derivatives
    //! @param x is the attributes/features/input to calculate on
    //! @param parameters is the values for weights, biases, etc
    //! @param activation/y is the output from the function, this will in turn
    //! be passed on to the backpropagation function
    virtual void calculateValues(ConstSpanD x,
                                 ConstSpanD parameters,
                                 SpanD y) = 0;

    //! Calculate the derivative of a node given parameters and output of
    //! previous calculation
    //! @param derivative output result from node
    virtual void backpropagate(ConstSpanD x,
                               ConstSpanD parameters,
                               ConstSpanD y,
                               ConstSpanD previousDerivative,
                               SpanD derivative) = 0;
};
