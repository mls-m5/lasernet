#pragma once

#ifdef __EMSCRIPTEN__

#include <filesystem>

namespace filesystem = std::__fs::filesystem;

#elif __has_include(<filesystem>)

#include <filesystem>

namespace filesystem = std::filesystem;

#else

#include <experimental/filesystem>

namespace filesystem = std::experimental::filesystem;

#endif
