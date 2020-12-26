#pragma once

#include "inode.h"
#include <algorithm>
#include <numeric>
#include <vector>

//! A network is a collection of nodes that can be used as a single node
class Graph : INode {
public:
    //! Used to partition a span into several smaller spans
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

    //! Used to define internal memory layout of data for storing all memory
    //! required for all layers
    struct NodeInfo {
        INode *node = nullptr;
        SpanPart input;

        //! Node that "parameters" is used to define layout of both parameters
        //! data and dEdw data
        SpanPart parameters;

        SpanPart output;
    };

    Graph() = default;

    Graph(std::vector<INode *> nodes) : _nodes() {
        size_t inputPosition = 0;
        size_t parameterPosition = 0;
        size_t outputPosition = 0;
        _nodes.reserve(nodes.size());
        for (auto node : nodes) {
            auto sizes = node->dataSize();
            //            auto parameterSize = node->parameterSize();
            //            auto outputSize = node->outputSize();

            _nodes.push_back({
                node,
                {inputPosition, sizes.input},
                {parameterPosition, sizes.parameters},
                {outputPosition, sizes.output},
            });

            inputPosition += sizes.input;
            parameterPosition += sizes.parameters;
            outputPosition += sizes.output;
        }
    }

    //! @see INode
    DataSize dataSize() override {
        auto sum = DataSize{0, 0, 0};

        return std::accumulate(_nodes.begin(),
                               _nodes.end(),
                               sum,
                               [](DataSize sum, NodeInfo &node) {
                                   sum.input += node.input.size;
                                   sum.parameters += node.parameters.size;
                                   sum.output += node.output.size;
                                   return sum;
                               });
    }

    //    // @see INode
    //    size_t parameterSize() override {
    //        return std::accumulate(
    //            _nodes.begin(), _nodes.end(), 0, [](size_t sum, NodeInfo
    //            &node) {
    //                return sum + node.parameters.size;
    //            });
    //    }

    //    // @see INode
    //    size_t outputSize() override {
    //        return std::accumulate(_nodes.begin(),
    //                               _nodes.end(),
    //                               size_t{0},
    //                               [](size_t sum, NodeInfo &node) {
    //                                   return sum + node.output.size;
    //                               });
    //    }

    ConstSpanD output(ConstSpanD data) override {
        auto &info = _nodes.back();

        return info.node->output(info.output(data));
    }

    // @see INode
    void calculateValues(CalculateArgs args) override {
        if (_nodes.empty()) {
            throw std::runtime_error("calculateValues on empty graph");
        }

        // First layer
        {
            auto &front = _nodes.front();
            front.node->calculateValues({
                .x = args.x,
                .parameters = front.parameters(args.parameters),
                .y = front.output(args.y),
            });
        }

        // The rest of the layers
        for (size_t i = 1; i < _nodes.size(); ++i) {
            auto &info = _nodes.at(i);
            info.node->calculateValues({
                .x = _nodes.at(i - 1).output(args.y),
                .parameters = info.parameters(args.parameters),
                .y = info.output(args.y),
            });
        }
    }

    // @see INode
    void backpropagate(BackpropagateArgs args) override {
        if (_nodes.size() < 2) {
            throw std::runtime_error(
                "backpropagate on graph with size smaller than 0");
        }

        // Back layer
        {
            auto &info = _nodes.back();

            info.node->backpropagate({
                .x = args.x,
                .parameters = info.parameters(args.parameters),
                .y = info.output(args.y),
                .dEdxPrev = args.dEdxPrev,
                .dEdx = args.dEdx,
                .dEdw = info.output(args.dEdw),
            });
        }

        // All layers except the first and last
        for (size_t i = _nodes.size() - 2; i != static_cast<size_t>(-1); --i) {
            auto &previous = _nodes.at(i);
            auto &info = _nodes.at(i - 1);

            info.node->backpropagate({
                .x = args.x,
                .parameters = info.parameters(args.parameters),
                .y = info.output(args.y),
                .dEdxPrev = previous.output(args.dEdw),
                .dEdx = info.input(args.dEdx),
                .dEdw = info.output(args.dEdw),
            });
        }
    }

private:
    std::vector<NodeInfo> _nodes;
};
