#pragma once

#include "network/inode.h"

class IOptimizer {
    virtual void step(INode *network) = 0;
};
