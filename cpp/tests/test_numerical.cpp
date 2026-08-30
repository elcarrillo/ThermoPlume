#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "thermoplume/numerical.hpp"


int main() {
    struct TestCase {
        double Omega;
        bool has_root;
        double expected;
    };

    const std::vector<TestCase> cases = {
        {
            0.00,
            true,
            0.7707277977523215
        },
        {
            0.05,
            true,
            0.7824825904995960
        },
        {
            0.15,
            true,
            0.8043373275568234
        },
        {
            0.30,
            false,
            0.0
        },
        {
            0.60,
            false,
            0.0
        },
    };

    std::cout
        << std::setprecision(16);

    for (const TestCase& test : cases) {
        thermoplume::Parameters p;

        p.Omega = test.Omega;

        const std::vector<double> roots =
            thermoplume::find_numerical_roots(
                p
            );

        std::cout
            << "Omega="
            << test.Omega
            << " roots=";

        for (double root : roots) {
            std::cout
                << root
                << " ";
        }

        std::cout << "\n";

        if (test.has_root) {
            if (
                roots.size() != 1
                || std::abs(
                    roots[0] - test.expected
                ) > 1e-10
            ) {
                return 1;
            }
        }

        else if (!roots.empty()) {
            return 1;
        }
    }

    std::cout
        << "C++ numerical roots agree\n";

    return 0;
}