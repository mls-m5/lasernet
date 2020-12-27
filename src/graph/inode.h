#pragma once

#include <cstdint>
#include <gsl/span>

using SpanD = gsl::span<double>;
using ConstSpanD = gsl::span<const double>;

class ICostFunction {
public:
    virtual ~ICostFunction() = default;

    virtual double cost(ConstSpanD activation, ConstSpanD expected) const = 0;

    //! Calculate the "derivative", ie the error to backpropagate
    //! @param activation is the activation of the last layer before this
    virtual void derive(ConstSpanD activation,
                        ConstSpanD expected,
                        SpanD derivative) const = 0;
};

class INode {
public:
    virtual ~INode() = default;

    struct DataSize {
        //! Data required to store derivative for input (dE/dx)
        //! If the size is larger than the layers input size, the extra data is
        //! assumed to be internal data only used for this layer
        //! This could be the case for layers that combines multiple nodes for
        //! example
        size_t input;

        //! Data required to store parameters and derivative to parameters
        //! (dE/dw)
        size_t parameters;

        //! Data required to store output/y/activation and output derivative
        size_t output;
    };

    virtual DataSize dataSize() const = 0;

    //! How much of the input data that is actual input
    //! Most casess uses all data ie returns "data"
    virtual ConstSpanD input(ConstSpanD data) const = 0;

    //! How much of the data is being sent to the next layer
    //! Returns the portion of the activation data that is going to be used
    //! If all is to be used, just return argument
    virtual ConstSpanD output(ConstSpanD data) const = 0;

    struct CalculateArgs {
        //! Input to the node
        ConstSpanD x;

        //! Current weights
        ConstSpanD parameters;

        //! Result of function call: calculated output
        SpanD y;
    };

    //! First step to get derivatives
    //! @param x is the attributes/features/input to calculate on
    //! @param parameters is the values for weights, biases, etc
    //! @param activation/y is the output from the function, this will in turn
    //! be passed on to the backpropagation function
    virtual void calculateValues(CalculateArgs) const = 0;

    //! Arguments for the backpropagate function
    //! Data is saved "somewhere else"
    //!
    //! Output is written to dEdx and dEdw
    struct BackpropagateArgs {
        //! Input used when calculating previously
        ConstSpanD x;

        //! Parameters/w is the value of weight and biases
        //! The part of the network that is trained during back propagation
        ConstSpanD parameters;

        //! Output, previously calculated
        ConstSpanD y;

        //! dEdx from previous layer
        //! ie previous dE/dx
        ConstSpanD dEdxPrev;

        //! Change in error caused by each input (dE/dx)
        //! Derivative that is associated with x/input
        SpanD dEdx;

        //! Change in error caused by each parameter (dE/dw)
        //! Derivativ that is associated with parameters/w
        SpanD dEdw;
    };

    //! Calculate the derivative of a node given parameters and output of
    //! previous calculation
    virtual void backpropagate(BackpropagateArgs) const = 0;
};
