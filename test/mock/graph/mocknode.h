#pragma once

#include "graph/inode.h"

#include "mls-unit-test/mock.h"

class MockNode : public INode {
public:
    MOCK_METHOD0(size_t, parameterSize, (), override);
    MOCK_METHOD0(size_t, activationSize, (), override);
    MOCK_METHOD1(ConstSpanD, output, (ConstSpanD data), override);
    MOCK_METHOD3(void,
                 calculateValues,
                 (ConstSpanD x, ConstSpanD parameters, SpanD activation),
                 override);
    MOCK_METHOD5(void,
                 backpropagate,
                 (ConstSpanD x,
                  ConstSpanD parameters,
                  ConstSpanD y,
                  ConstSpanD previousDerivative,
                  SpanD derivative),
                 override);
};
