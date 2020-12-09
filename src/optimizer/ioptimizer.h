#pragma once

#include "graph/inode.h"

class IOptimizer {
public:
    virtual void applyDerivative(ConstSpanD derivative,
                                 double learningRate,
                                 SpanD parameters) = 0;
};
