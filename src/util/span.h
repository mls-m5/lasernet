#pragma once

#include <cstdint>
#include <exception>

template <typename T>
struct Span {
    T *_begin;
    T *_end;

    Span(T *begin, T *end) : _begin(begin), _end(end) {}

    T *begin() {
        return _begin;
    }

    T *begin() const {
        return _begin;
    }

    T *end() {
        return _end;
    }

    T *end() const {
        return _end;
    }

    T &front() {
        return *begin();
    }

    T &back() {
        return *(_end - 1);
    }

    T &front() const {
        return *begin();
    }

    T &back() const {
        return *(_end - 1);
    }

    T &at(size_t index) {
        if (index > (_end - _begin)) {
            std::out_of_range("out of range in Span");
        }
        return *(_begin + index);
    }

    T &at(size_t index) const {
        if (index > (_end - _begin)) {
            std::out_of_range("out of range in Span");
        }
        return *(_begin + index);
    }
};

using SpanD = Span<double>;
using SpanF = Span<float>;
