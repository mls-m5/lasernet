#include "layers/linearcost.h"
#include "layers/singlestraightline.h"
#include "optimizer/gradientdescent.h"
#include "optimizer/trainer.h"
#include <iostream>

int main(int, char **) {
    // Test workflow:

    // LoadInput
    // auto data = load("mnist-for-example")
    auto data = Dataset{}; // Todo: Fill with actual data

    auto optimizer = GradientDescent{};

    auto trainer = Trainer{optimizer, data, 1};

    auto node = SingleStraightLine{}; // Or network in the future

    auto cost = LinearCost{};

    for (size_t i = 0; i < 100; ++i) {
        trainer.step(node, cost);

        // std::cout << trainer.cost() << std::endl;
    }

    return 0;
}
