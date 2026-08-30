import numpy as np
from scipy.optimize import brentq

from .parameters import ThermoPlumeParameters


def numerical_thermo_state(xi, p):
    # mixture fractions
    omega = p.Omega * xi
    Y_air = 1.0 - (1.0 + p.Omega) * xi

    # excess enthalpy relative to saturation
    H_i = (
        xi * p.Cp_m * (p.T_m - p.T_sat)
        + Y_air * p.Cp_air * (p.T_air - p.T_sat)
        + omega * p.Cp_l * (p.T_w0 - p.T_sat)
    )

    latent_demand = omega * p.L_v

    # regime A no boiling
    if H_i <= 0.0:
        regime = "A"

        R_mix = (
            xi * p.n * p.R_g
            + Y_air * p.R_air
        )

        numerator = (
            xi * p.Cp_m * p.T_m
            + Y_air * p.Cp_air * p.T_air
            + omega * p.Cp_l * p.T_w0
        )

        denominator = (
            xi * p.Cp_m
            + Y_air * p.Cp_air
            + omega * p.Cp_l
        )

        T = numerator / denominator

    # regime B partial boiling
    elif H_i < latent_demand:
        regime = "B"

        omega_vapor = H_i / p.L_v

        R_mix = (
            xi * p.n * p.R_g
            + Y_air * p.R_air
            + omega_vapor * p.R_w
        )

        T = p.T_sat

    # regime C complete vaporization
    else:
        regime = "C"

        R_mix = (
            xi * p.n * p.R_g
            + omega * p.R_w
            + Y_air * p.R_air
        )

        denominator = (
            xi * p.Cp_m
            + Y_air * p.Cp_air
            + omega * p.Cp_w
        )

        T = (
            p.T_sat
            + (H_i - latent_demand) / denominator
        )

    return {
        "T": T,
        "R_mix": R_mix,
        "regime": regime,
        "H_i": H_i,
        "Y_air": Y_air,
        "omega_eff": omega,
    }


def neutral_buoyancy_residual(xi, p):
    state = numerical_thermo_state(xi, p)

    return (
        state["R_mix"] * state["T"]
        - p.R_air * p.T_air
    )


def find_numerical_roots(
    p,
    n_scan=3000,
    xi_min=1e-8,
    root_tol=1e-12,
):
    xi_max = 1.0 / (1.0 + p.Omega)

    if xi_max <= xi_min:
        return []

    xi_upper = xi_max * (1.0 - 1e-12)

    xi_grid = np.linspace(
        xi_min,
        xi_upper,
        n_scan,
    )

    # thermodynamic regime boundaries
    H0 = p.Cp_air * (
        p.T_air - p.T_sat
    )

    H1 = (
        p.Cp_m * (p.T_m - p.T_sat)
        - (1.0 + p.Omega)
        * p.Cp_air
        * (p.T_air - p.T_sat)
        + p.Omega
        * p.Cp_l
        * (p.T_w0 - p.T_sat)
    )

    extra_points = []

    # A/B boundary
    if abs(H1) > 1e-14:
        xi_AB = -H0 / H1

        if xi_min < xi_AB < xi_upper:
            extra_points.append(xi_AB)

    # B/C boundary
    denom_BC = H1 - p.Omega * p.L_v

    if abs(denom_BC) > 1e-14:
        xi_BC = -H0 / denom_BC

        if xi_min < xi_BC < xi_upper:
            extra_points.append(xi_BC)

    if extra_points:
        xi_grid = np.sort(
            np.unique(
                np.concatenate([
                    xi_grid,
                    np.asarray(extra_points),
                ])
            )
        )

    F = np.asarray([
        neutral_buoyancy_residual(xi, p)
        for xi in xi_grid
    ])

    roots = []

    for i in range(len(xi_grid) - 1):
        x1 = xi_grid[i]
        x2 = xi_grid[i + 1]

        f1 = F[i]
        f2 = F[i + 1]

        if not (
            np.isfinite(f1)
            and np.isfinite(f2)
        ):
            continue

        if abs(f1) < 1e-10:
            roots.append(x1)

        if f1 * f2 < 0.0:
            root = brentq(
                neutral_buoyancy_residual,
                x1,
                x2,
                args=(p,),
                xtol=root_tol,
                rtol=root_tol,
                maxiter=200,
            )

            roots.append(root)

    if abs(F[-1]) < 1e-10:
        roots.append(xi_grid[-1])

    roots = sorted(roots)

    unique_roots = []

    for root in roots:
        if root <= 1e-7:
            continue

        if not unique_roots:
            unique_roots.append(root)

        elif abs(root - unique_roots[-1]) > 1e-8:
            unique_roots.append(root)

    return unique_roots