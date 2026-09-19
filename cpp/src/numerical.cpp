#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "thermoplume/numerical.hpp"
#include "thermoplume/thermodynamics.hpp"


namespace {

using ResidualFunction = double (*)(
    double,
    const thermoplume::Parameters&
);

double brent_root(
    ResidualFunction residual,
    const thermoplume::Parameters& p,
    double x1,
    double x2,
    double xtol,
    double rtol,
    int maxiter
) {
    double a = x1;
    double b = x2;
    double c = x2;

    double fa =
        residual(
            a,
            p
        );

    double fb =
        residual(
            b,
            p
        );

    double fc = fb;

    // root already on bracket endpoint
    if (fa == 0.0) {
        return a;
    }

    if (fb == 0.0) {
        return b;
    }

    // Brent method requires a sign-changing bracket
    if (fa * fb > 0.0) {
        throw std::runtime_error(
            "Brent root is not bracketed"
        );
    }

    double d = b - a;
    double e = d;

    for (int iter = 0; iter < maxiter; ++iter) {
        // maintain sign-changing bracket
        if (
            (fb > 0.0 && fc > 0.0)
            || (fb < 0.0 && fc < 0.0)
        ) {
            c = a;
            fc = fa;

            d = b - a;
            e = d;
        }

        // keep b as best current estimate
        if (std::abs(fc) < std::abs(fb)) {
            a = b;
            b = c;
            c = a;

            fa = fb;
            fb = fc;
            fc = fa;
        }

        const double tol1 =
            0.5 * xtol
            + rtol * std::abs(b);

        const double midpoint =
            0.5 * (c - b);

        // converged
        if (
            std::abs(midpoint) <= tol1
            || fb == 0.0
        ) {
            return b;
        }

        // interpolation when bracket geometry allows
        if (
            std::abs(e) >= tol1
            && std::abs(fa) > std::abs(fb)
        ) {
            const double s = fb / fa;

            double q;
            double r;
            double step;

            // secant step
            if (a == c) {
                step =
                    2.0 * midpoint * s;

                q =
                    1.0 - s;
            }

            // inverse quadratic interpolation
            else {
                q = fa / fc;
                r = fb / fc;

                step =
                    s
                    * (
                        2.0
                        * midpoint
                        * q
                        * (q - r)
                        - (b - a)
                        * (r - 1.0)
                    );

                q =
                    (q - 1.0)
                    * (r - 1.0)
                    * (s - 1.0);
            }

            if (step > 0.0) {
                q = -q;
            }

            step = std::abs(step);

            const double min1 =
                3.0
                * midpoint
                * q
                - std::abs(tol1 * q);

            const double min2 =
                std::abs(e * q);

            // accept interpolation step
            if (
                q != 0.0
                && 2.0 * step
                < std::min(min1, min2)
            ) {
                e = d;
                d = step / q;
            }

            // fall back to bisection
            else {
                d = midpoint;
                e = d;
            }
        }

        // bisection when interpolation is unsafe
        else {
            d = midpoint;
            e = d;
        }

        a = b;
        fa = fb;

        // update root estimate
        if (std::abs(d) > tol1) {
            b += d;
        }

        else {
            b += std::copysign(
                tol1,
                midpoint
            );
        }

        fb =
            residual(
                b,
                p
            );
    }

    throw std::runtime_error(
        "Brent root solver exceeded max iterations"
    );
}

}  // namespace


namespace thermoplume {

ThermoState numerical_thermo_state(
    double xi,
    const Parameters& p
) {
    // mixture mass fractions
    const double omega =
        p.Omega * xi;

    const double Y_air =
        1.0 - (1.0 + p.Omega) * xi;

    // available enthalpy relative to T_sat
    const double H_i =
        xi * p.Cp_m * (p.T_m - p.T_sat)
        + Y_air * p.Cp_air * (p.T_air - p.T_sat)
        + omega * p.Cp_l * (p.T_w0 - p.T_sat);

    // energy needed to vaporize all external water
    const double latent_demand =
        omega * p.L_v;

    double T;
    double R_mix;
    std::string regime;

    // regime A no boiling H_i <= 0
    if (H_i <= 0.0) {
        regime = "A";

        // gas constant from volcanic gas and entrained air
        R_mix =
            xi * p.n * p.R_g
            + Y_air * p.R_air;

        // sensible heat balance below T_sat
        const double numerator =
            xi * p.Cp_m * p.T_m
            + Y_air * p.Cp_air * p.T_air
            + omega * p.Cp_l * p.T_w0;

        const double denominator =
            xi * p.Cp_m
            + Y_air * p.Cp_air
            + omega * p.Cp_l;

        T =
            numerator / denominator;
    }

    // regime B partial boiling 0 < H_i < omega L_v
    else if (H_i < latent_demand) {
        regime = "B";

        // vaporized water from available latent energy
        const double omega_vapor =
            H_i / p.L_v;

        // gas constant includes partial water vapor
        R_mix =
            xi * p.n * p.R_g
            + Y_air * p.R_air
            + omega_vapor * p.R_w;

        // phase change holds mixture at saturation
        T = p.T_sat;
    }

    // regime C complete vaporization H_i >= omega L_v
    else {
        regime = "C";

        // all external water contributes as vapor
        R_mix =
            xi * p.n * p.R_g
            + omega * p.R_w
            + Y_air * p.R_air;

        // heat capacity after complete vaporization
        const double denominator =
            xi * p.Cp_m
            + Y_air * p.Cp_air
            + omega * p.Cp_w;

        // remaining enthalpy superheats mixture above T_sat
        T =
            p.T_sat
            + (
                H_i - latent_demand
            ) / denominator;
    }

    return {
        T,
        R_mix,
        regime,
        H_i,
        Y_air,
        omega,
    };
}


double neutral_buoyancy_residual(
    double xi,
    const Parameters& p
) {
    const ThermoState state =
        numerical_thermo_state(
            xi,
            p
        );

    // zero at neutral buoyancy
    return (
        state.R_mix * state.T
        - p.R_air * p.T_air
    );
}


ThermoState vapor_thermo_state(
    double xi,
    const Parameters& p
) {
    const double omega = p.Omega * xi;
    const double Y_air = 1.0 - (1.0 + p.Omega) * xi;

    const double R_mix =
        xi * p.n * p.R_g
        + omega * p.R_w
        + Y_air * p.R_air;

    const double numerator =
        xi * p.Cp_m * p.T_m
        + omega * p.Cp_w * p.T_w
        + Y_air * p.Cp_air * p.T_air;

    const double denominator =
        xi * p.Cp_m
        + omega * p.Cp_w
        + Y_air * p.Cp_air;

    const double T = numerator / denominator;

    return {
        T,
        R_mix,
        "vapor",
        0.0,
        Y_air,
        omega,
    };
}


double vapor_neutral_buoyancy_residual(
    double xi,
    const Parameters& p
) {
    const ThermoState state =
        vapor_thermo_state(xi, p);

    return (
        state.R_mix * state.T
        - p.R_air * p.T_air
    );
}

std::vector<double> find_numerical_roots(
    const Parameters& p,
    int n_scan,
    double xi_min,
    double root_tol
) {
    if (n_scan < 2) {
        throw std::invalid_argument(
            "n_scan must be at least 2"
        );
    }

    // max xi from nonnegative air fraction
    const double xi_max =
        1.0 / (1.0 + p.Omega);

    if (xi_max <= xi_min) {
        return {};
    }

    // avoid exact Y_air = 0 endpoint
    const double xi_upper =
        xi_max * (1.0 - 1e-12);

    std::vector<double> xi_grid;

    xi_grid.reserve(
        static_cast<std::size_t>(n_scan) + 2
    );

    // base scan for sign changes
    for (int i = 0; i < n_scan; ++i) {
        const double fraction =
            static_cast<double>(i)
            / static_cast<double>(n_scan - 1);

        xi_grid.push_back(
            xi_min
            + fraction
            * (xi_upper - xi_min)
        );
    }

    // exact thermodynamic regime transitions
    const auto [
        xi_AB,
        xi_BC
    ] = regime_boundaries(p);

    // onset of boiling
    if (
        std::isfinite(xi_AB)
        && xi_min < xi_AB
        && xi_AB < xi_upper
    ) {
        xi_grid.push_back(xi_AB);
    }

    // complete vaporization
    if (
        std::isfinite(xi_BC)
        && xi_min < xi_BC
        && xi_BC < xi_upper
    ) {
        xi_grid.push_back(xi_BC);
    }

    // sort and remove duplicate scan locations
    std::sort(
        xi_grid.begin(),
        xi_grid.end()
    );

    xi_grid.erase(
        std::unique(
            xi_grid.begin(),
            xi_grid.end()
        ),
        xi_grid.end()
    );

    // evaluate neutral buoyancy residual
    std::vector<double> residuals;

    residuals.reserve(
        xi_grid.size()
    );

    for (double xi : xi_grid) {
        residuals.push_back(
            neutral_buoyancy_residual(
                xi,
                p
            )
        );
    }

    std::vector<double> roots;

    // bracket sign changes and refine with Brent method
    for (
        std::size_t i = 0;
        i + 1 < xi_grid.size();
        ++i
    ) {
        const double x1 =
            xi_grid[i];

        const double x2 =
            xi_grid[i + 1];

        const double f1 =
            residuals[i];

        const double f2 =
            residuals[i + 1];

        // skip invalid states
        if (
            !std::isfinite(f1)
            || !std::isfinite(f2)
        ) {
            continue;
        }

        // root directly on scan point
        if (std::abs(f1) < 1e-10) {
            roots.push_back(x1);
        }

        // bracketed root
        if (f1 * f2 < 0.0) {
            roots.push_back(
                brent_root(
                    neutral_buoyancy_residual,
                    p,
                    x1,
                    x2,
                    root_tol,
                    root_tol,
                    200
                )
            );
        }
    }

    // check final scan point
    if (
        std::abs(
            residuals.back()
        ) < 1e-10
    ) {
        roots.push_back(
            xi_grid.back()
        );
    }

    std::sort(
        roots.begin(),
        roots.end()
    );

    // remove xi near zero and duplicate roots
    std::vector<double> unique_roots;

    for (double root : roots) {
        if (root <= 1e-7) {
            continue;
        }

        if (
            unique_roots.empty()
            || std::abs(
                root - unique_roots.back()
            ) > 1e-8
        ) {
            unique_roots.push_back(root);
        }
    }

    return unique_roots;
}


std::vector<double> find_vapor_numerical_roots(
    const Parameters& p,
    int n_scan,
    double xi_min,
    double root_tol
) {
    if (n_scan < 2) {
        throw std::invalid_argument(
            "n_scan must be at least 2"
        );
    }

    const double xi_max = 1.0 / (1.0 + p.Omega);

    if (xi_max <= xi_min) {
        return {};
    }

    const double xi_upper = xi_max * (1.0 - 1e-12);

    std::vector<double> xi_grid;
    xi_grid.reserve(static_cast<std::size_t>(n_scan));

    for (int i = 0; i < n_scan; ++i) {
        const double fraction =
            static_cast<double>(i)
            / static_cast<double>(n_scan - 1);

        xi_grid.push_back(
            xi_min + fraction * (xi_upper - xi_min)
        );
    }

    std::vector<double> residuals;
    residuals.reserve(xi_grid.size());

    for (double xi : xi_grid) {
        residuals.push_back(
            vapor_neutral_buoyancy_residual(xi, p)
        );
    }

    std::vector<double> roots;

    for (std::size_t i = 0; i + 1 < xi_grid.size(); ++i) {
        const double x1 = xi_grid[i];
        const double x2 = xi_grid[i + 1];
        const double f1 = residuals[i];
        const double f2 = residuals[i + 1];

        if (!std::isfinite(f1) || !std::isfinite(f2)) {
            continue;
        }

        if (std::abs(f1) < 1e-10) {
            roots.push_back(x1);
        }

        if (f1 * f2 < 0.0) {
            roots.push_back(
                brent_root(
                    vapor_neutral_buoyancy_residual,
                    p,
                    x1,
                    x2,
                    root_tol,
                    root_tol,
                    200
                )
            );
        }
    }

    if (
        !residuals.empty()
        && std::abs(residuals.back()) < 1e-10
    ) {
        roots.push_back(xi_grid.back());
    }

    std::sort(roots.begin(), roots.end());

    std::vector<double> unique_roots;

    for (double root : roots) {
        if (root <= 1e-7) {
            continue;
        }

        if (
            unique_roots.empty()
            || std::abs(root - unique_roots.back()) > 1e-8
        ) {
            unique_roots.push_back(root);
        }
    }

    return unique_roots;
}

}  // namespace thermoplume