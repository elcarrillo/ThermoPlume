#include <cmath>
#include <limits>

#include "thermoplume/thermodynamics.hpp"


namespace thermoplume {

std::pair<double, double> regime_boundaries(
    const Parameters& p
) {
    // H_i = H0 + H1 xi
    const double H0 =
        p.Cp_air * (
            p.T_air - p.T_sat
        );

    const double H1 =
        p.Cp_m * (p.T_m - p.T_sat)
        - (1.0 + p.Omega)
        * p.Cp_air
        * (p.T_air - p.T_sat)
        + p.Omega
        * p.Cp_l
        * (p.T_w0 - p.T_sat);

    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    double xi_AB = nan;
    double xi_BC = nan;

    // onset of boiling H_i = 0
    if (std::abs(H1) > 1e-14) {
        xi_AB = -H0 / H1;
    }

    // complete vaporization H_i = omega L_v
    const double denom_BC =
        H1 - p.Omega * p.L_v;

    if (std::abs(denom_BC) > 1e-14) {
        xi_BC = -H0 / denom_BC;
    }

    return {
        xi_AB,
        xi_BC,
    };
}

}  // namespace thermoplume