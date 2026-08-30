#pragma once

#include <string>
#include <vector>

#include "thermoplume/parameters.hpp"


namespace thermoplume {

struct ThermoState {
    // thermodynamic state at specified erupted-material fraction xi
    double T;
    double R_mix;
    std::string regime;
    double H_i;
    double Y_air;
    double omega_eff;
};


ThermoState numerical_thermo_state(
    double xi,
    const Parameters& p
);


double neutral_buoyancy_residual(
    double xi,
    const Parameters& p
);


std::vector<double> find_numerical_roots(
    const Parameters& p,
    int n_scan = 3000,
    double xi_min = 1e-8,
    double root_tol = 1e-12
);

}  // namespace thermoplume