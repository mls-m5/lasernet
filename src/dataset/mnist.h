#pragma once

#include "dataset/dataset.h"
#include "files/filesystem.h"
#include "msl/range.h"
#include "types/spand.h"
#include <algorithm>
#include <fstream>

//! For mor info see
//! http://yann.lecun.com/exdb/mnist/
CombinedDataset loadMnist(filesystem::path directory) {
    auto trainImagesPath = directory / "train-images-idx3-ubyte";
    auto trainLabelsPath = directory / "train-labels-idx1-ubyte";

    auto testImagesPath = directory / "t10k-images-idx3-ubyte";
    auto testLabelsPath = directory / "t10k-labels-idx1-ubyte";

    auto loadDataset = [](filesystem::path imagePath,
                          filesystem::path labelPath) {
        struct Header {
            uint32_t magicNumber = 0;
            uint32_t numberOfImages = 0;
        };

        auto labelFile = std::ifstream{labelPath, std::ios::binary};
        auto imageFile = std::ifstream{imagePath, std::ios::binary};

        auto readInt = [](std::istream &stream) {
            auto ret = uint32_t{0};

            auto data = std::array<uint8_t, 4>{};
            stream.read(reinterpret_cast<char *>(data.data()), data.size());

            for (auto c : data) {
                ret <<= 8;
                ret += c;
            }

            return ret;
        };

        if (!imageFile.is_open()) {
            throw std::runtime_error("could not open " + imagePath.string());
        }
        if (!labelFile.is_open()) {
            throw std::runtime_error("could not open " + labelPath.string());
        }

        auto labelHeader = Header{
            .magicNumber = readInt(labelFile),
            .numberOfImages = readInt(labelFile),
        };

        auto imageHeader = Header{
            .magicNumber = readInt(imageFile),
            .numberOfImages = readInt(imageFile),
        };

        const auto width = readInt(imageFile);
        const auto height = readInt(imageFile);

        if (labelHeader.magicNumber != 2049) {
            throw std::runtime_error("wrong magick number in " +
                                     labelPath.string());
        }

        if (imageHeader.magicNumber != 2051) {
            throw std::runtime_error("wrong magick number in " +
                                     imagePath.string());
        }

        if (labelHeader.numberOfImages != imageHeader.numberOfImages) {
            throw std::runtime_error("number of images in mnist dataset " +
                                     imagePath.string() + " and " +
                                     labelPath.string() + " does not match");
        }

        auto rawLabels = std::vector<char>(labelHeader.numberOfImages);

        if (width != 28 && height != 28) {
            throw std::runtime_error("invalid dimesions when loading mnist");
        }

        const auto imageSize = width * height;

        std::vector<uint8_t> data(imageSize * imageHeader.numberOfImages);

        imageFile.read(reinterpret_cast<char *>(data.data()), data.size());
        std::vector<double> ddata(data.size());

        std::transform(data.begin(), data.end(), ddata.begin(), [](uint8_t d) {
            return static_cast<double>(d) / 255.;
        });

        data.clear();

        std::vector<Dataset::DataPair> pairs(imageHeader.numberOfImages);

        for (auto i : msl::range(labelHeader.numberOfImages)) {
            auto &pair = pairs.at(i);

            pair.x.assign(ddata.data() + i * imageSize,
                          ddata.data() + (i + 1) * imageSize);

            // Convert labels to one hot vector
            pair.y.resize(10, 0);
            pair.y.at(rawLabels.at(i)) = 1;
        }

        return Dataset{std::move(pairs)};
    };

    return {
        .train = loadDataset(trainImagesPath, trainLabelsPath),
        .test = loadDataset(testImagesPath, testLabelsPath),
    };
}

void printMnistImage(std::ostream &stream, ConstSpanD data) {
    auto drawPixel = [&stream](double pixel) {
        // Add more scale steps in the future
        if (pixel > .5) {
            stream << 'x';
        }
        else {
            stream << ' ';
        }
    };

    auto it = data.begin();
    for (auto y : msl::range(28)) {
        (void)y;
        for (auto x : msl::range(28)) {
            (void)x;
            drawPixel(*it);
            ++it;
        }

        stream << "\n";
    }
}
