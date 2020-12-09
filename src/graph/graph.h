#pragma once

#include "inode.h"
#include <algorithm>
#include <numeric>
#include <vector>

//! A network is a collection of nodes that can be used as a single node
class Graph : INode {
public:
    //! Partition a span into several smaller spans
    struct SpanPart {
        size_t offset = 0;
        size_t size = 0;

        SpanD operator()(SpanD span) const {
            return {span.data() + offset, span.data() + offset + size};
        }

        ConstSpanD operator()(ConstSpanD span) const {
            return {span.data() + offset, span.data() + offset + size};
        }
    };

    struct NodeInfo {
        INode *node = nullptr;
        SpanPart parameters;
        SpanPart activation;
    };

    Graph() = default;

    Graph(std::vector<INode *> nodes) : _nodes() {
        size_t parameterPosition = 0;
        size_t activationPosition = 0;
        _nodes.reserve(nodes.size());
        for (auto node : nodes) {
            auto parameterSize = node->parameterSize();
            auto activationSize = node->activationSize();

            _nodes.push_back({
                node,
                {parameterPosition, parameterSize},
                {activationPosition, activationSize},
            });

            parameterPosition += parameterSize;
            activationPosition += activationSize;
        }
    }

    // @see INode
    size_t parameterSize() override {
        return std::accumulate(
            _nodes.begin(), _nodes.end(), 0, [](size_t sum, NodeInfo &node) {
                return sum + node.parameters.size;
            });
    }

    // @see INode
    size_t activationSize() override {
        return std::accumulate(_nodes.begin(),
                               _nodes.end(),
                               size_t{0},
                               [](size_t sum, NodeInfo &node) {
                                   return sum + node.activation.size;
                               });
    }

    ConstSpanD output(ConstSpanD data) override {
        auto &info = _nodes.back();

        return info.node->output(info.activation(data));
    }

    // @see INode
    void calculateValues(ConstSpanD x,
                         ConstSpanD parameters,
                         SpanD activation) override {
        if (_nodes.empty()) {
            throw std::runtime_error("calculateValues on empty graph");
        }

        // First layer
        {
            auto &front = _nodes.front();
            front.node->calculateValues(
                x, front.parameters(parameters), front.activation(activation));
        }

        // The rest of the layers
        for (size_t i = 1; i < _nodes.size(); ++i) {
            auto &info = _nodes.at(i);
            info.node->calculateValues(_nodes.at(i - 1).activation(activation),
                                       info.parameters(parameters),
                                       info.activation(activation));
        }
    }

    // @see INode
    void backpropagate(ConstSpanD x,
                       ConstSpanD parameters,
                       ConstSpanD activation,
                       ConstSpanD previousDerivative,
                       SpanD derivative) override {
        if (_nodes.size() < 2) {
            throw std::runtime_error(
                "backpropagate on graph with size smaller than 0");
        }

        // Back layer
        {
            auto &info = _nodes.back();

            info.node->backpropagate(x,
                                     info.parameters(parameters),
                                     info.activation(activation),
                                     previousDerivative,
                                     info.activation(derivative));
        }

        // All layers except the first and last
        for (size_t i = _nodes.size() - 2; i != static_cast<size_t>(-1); --i) {
            auto &previous = _nodes.at(i);
            auto &info = _nodes.at(i - 1);

            info.node->backpropagate(x,
                                     info.parameters(parameters),
                                     info.activation(activation),
                                     previous.activation(derivative),
                                     info.activation(derivative));
        }
    }

private:
    std::vector<NodeInfo> _nodes;
};
