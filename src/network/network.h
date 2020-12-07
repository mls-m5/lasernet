#pragma once

#include "inode.h"

class Network : INode {
    Network() {}

    // INode interface
private:
    void calculateValues(const Span<double> parameters, SpanD activation);
    void backpropagate(const SpanD parameters,
                       const SpanD activation,
                       const SpanD previousDerivative,
                       SpanD derivative);
    size_t parameterSize();
    size_t activationSize();
};
