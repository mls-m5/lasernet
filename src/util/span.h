#pragma once

#include <cstdint>
#include <execution>

template <typename T>
struct Span {
    T *_begin;
    T *_end;

    Span(T *begin, T *end) : _begin(begin), _end(end) {}

    auto begin() {
        return _begin;
    }

    auto begin() const {
        return _begin;
    }

    auto end() {
        return _end;
    }

    auto end() const {
        return _end;
    }

    auto at(size_t index) {
        if (index > (_end - _begin)) {
            std::out_of_range("out of range in Span");
        }
        return _begin + index;
    }

    auto at(size_t index) const {
        if (index > (_end - _begin)) {
            std::out_of_range("out of range in Span");
        }
        return _begin + index;
    }
};

using SpanD = Span<double>;
using SpanF = Span<float>;
