#pragma once

#include <utility>

#include "thermoplume/parameters.hpp"


namespace thermoplume {

std::pair<double, double> regime_boundaries(
    const Parameters& p
);

}  // namespace thermoplume