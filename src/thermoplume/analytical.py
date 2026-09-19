# analytical neutral buoyancy solutions for dry and water-loaded plumes
#    builds closed-form roots for thermodynamic regimes A B and C


import numpy as np

from .parameters import ThermoPlumeParameters

from .thermodynamics import (
    available_enthalpy,
    fractions_from_xi,
    in_bounds,
    valid_air_fraction,
)

def quadratic_roots(a, b, c, tol=1e-14):
    # solve quadratic or linear equation
    if abs(a) < tol:
        if abs(b) < tol:
            return []

        return [-c / b]

    disc = b * b - 4.0 * a * c

    # no real roots
    if disc < -tol:
        return []

    # repeated root
    if abs(disc) <= tol:
        return [-b / (2.0 * a)]

    # two real roots
    sqrt_disc = np.sqrt(max(disc, 0.0))

    return [
        (-b + sqrt_disc) / (2.0 * a),
        (-b - sqrt_disc) / (2.0 * a),
    ]


def xi_crit_dry(p):
    # analytical critical erupted-material fraction without external water
    numerator = (
        p.R_air * p.Cp_m * (p.T_m - p.T_air)
        - p.Cp_air
        * p.T_air
        * (p.R_air - p.n * p.R_g)
    )

    denominator = (
        (p.R_air - p.n * p.R_g)
        * (p.Cp_m * p.T_m - p.Cp_air * p.T_air)
    )

    if abs(denominator) < 1e-14:
        return np.nan

    return numerator / denominator


def derived_Omega_terms(p):
    # reusable coefficients for Omega = m_w / m_erupted analytical equations
    O = p.Omega

    # constant terms at xi = 0
    N0 = p.Cp_air * p.T_air
    D0 = p.Cp_air
    H0 = p.Cp_air * (p.T_air - p.T_sat)

    # regime A sensible heat numerator and denominator slopes
    N1_A = (
        p.Cp_m * p.T_m
        - (1.0 + O) * p.Cp_air * p.T_air
        + O * p.Cp_l * p.T_w0
    )

    D1_A = (
        p.Cp_m
        - (1.0 + O) * p.Cp_air
        + O * p.Cp_l
    )

    # regime C vapor heat-capacity slope
    D1_C = (
        p.Cp_m
        - (1.0 + O) * p.Cp_air
        + O * p.Cp_w
    )

    # available enthalpy slope relative to saturation
    H1 = (
        p.Cp_m * (p.T_m - p.T_sat)
        - (1.0 + O)
        * p.Cp_air
        * (p.T_air - p.T_sat)
        + O
        * p.Cp_l
        * (p.T_w0 - p.T_sat)
    )

    # gas constant slopes for regimes A B and C
    R1_A = (
        p.n * p.R_g
        - (1.0 + O) * p.R_air
    )

    R1_B_base = R1_A

    R1_C = (
        p.n * p.R_g
        + O * p.R_w
        - (1.0 + O) * p.R_air
    )

    return {
        "N0": N0,
        "D0": D0,
        "H0": H0,
        "N1_A": N1_A,
        "D1_A": D1_A,
        "D1_C": D1_C,
        "H1": H1,
        "R1_A": R1_A,
        "R1_B_base": R1_B_base,
        "R1_C": R1_C,
    }


# ======================================================================
# analytical neutral buoyancy roots for thermodynamic regimes A B and C
# regimes A and C reduce to quadratic equations in xi
# regime B reduces to a linear equation because T = T_sat
# ======================================================================

def regime_A_coeffs(p):
    # regime A no boiling
    # external water remains liquid and acts as a sensible heat sink
    q = derived_Omega_terms(p)

    # neutral buoyancy
    # (R_air + R1_A xi)(N0 + N1_A xi)
    # = R_air T_air (D0 + D1_A xi)

    # constant terms cancel leaving xi(a xi + b) = 0
    a = q["R1_A"] * q["N1_A"]

    b = (
        p.R_air * q["N1_A"]
        + q["R1_A"] * q["N0"]
        - p.R_air * p.T_air * q["D1_A"]
    )

    c = 0.0

    return a, b, c


def xi_crit_B(p):
    # regime B partial boiling at T_sat
    # available enthalpy vaporizes only part of the external water
    q = derived_Omega_terms(p)

    # neutral buoyancy reduces to a linear equation in xi
    denominator = (
        q["R1_B_base"]
        + (p.R_w / p.L_v) * q["H1"]
    ) * p.T_sat

    numerator = (
        p.R_air * p.T_air
        - (
            p.R_air
            + (p.R_w / p.L_v) * q["H0"]
        )
        * p.T_sat
    )

    if abs(denominator) < 1e-14:
        return np.nan

    return numerator / denominator


def regime_C_coeffs(p):
    # regime C complete vaporization
    # remaining enthalpy after vaporization superheats the mixture
    q = derived_Omega_terms(p)

    # temperature numerator B0 + B1 xi
    B0 = (
        p.T_sat * q["D0"]
        + q["H0"]
    )

    B1 = (
        p.T_sat * q["D1_C"]
        + q["H1"]
        - p.Omega * p.L_v
    )

    # neutral buoyancy
    # (R_air + R1_C xi)(B0 + B1 xi)
    # = R_air T_air (D0 + D1_C xi)

    # constant terms cancel leaving xi(a xi + b) = 0
    a = q["R1_C"] * B1

    b = (
        p.R_air * B1
        + q["R1_C"] * B0
        - p.R_air * p.T_air * q["D1_C"]
    )

    c = 0.0

    return a, b, c

# ======================================================================
# checks whether each analytical root satisfies its thermodynamic regime
# roots must also satisfy mixture composition and entrained-air constraints
# ======================================================================

def valid_regime_A(xi, p):
    # reject roots outside physical mixture composition
    if not in_bounds(xi, p):
        return False

    # reject roots below minimum entrained-air fraction
    if not valid_air_fraction(xi, p):
        return False

    H_i = available_enthalpy(xi, p)

    omega, _ = fractions_from_xi(xi, p)
    latent_demand = omega * p.L_v

    # tolerance near regime boundaries
    tol = 1e-10 * max(
        1.0,
        abs(H_i),
        abs(latent_demand),
    )

    # no boiling
    return H_i <= tol


def valid_regime_B(xi, p):
    # reject roots outside physical mixture composition
    if not in_bounds(xi, p):
        return False

    # reject roots below minimum entrained-air fraction
    if not valid_air_fraction(xi, p):
        return False

    omega, _ = fractions_from_xi(xi, p)

    # partial boiling requires external water
    if omega <= 0.0:
        return False

    H_i = available_enthalpy(xi, p)
    latent_demand = omega * p.L_v

    # tolerance near regime boundaries
    tol = 1e-10 * max(
        1.0,
        abs(H_i),
        abs(latent_demand),
    )

    # enough energy to boil some but not all water
    return (
        H_i > tol
        and H_i < latent_demand - tol
    )


def valid_regime_C(xi, p):
    # reject roots outside physical mixture composition
    if not in_bounds(xi, p):
        return False

    # reject roots below minimum entrained-air fraction
    if not valid_air_fraction(xi, p):
        return False

    omega, _ = fractions_from_xi(xi, p)

    # complete vaporization requires external water
    if omega <= 0.0:
        return False

    H_i = available_enthalpy(xi, p)
    latent_demand = omega * p.L_v

    # tolerance near regime boundaries
    tol = 1e-10 * max(
        1.0,
        abs(H_i),
        abs(latent_demand),
    )

    # enough energy to vaporize all external water
    return H_i >= latent_demand - tol

# ======================================================================
# finds all physically admissible analytical roots
# selects the root nearest the previous solution to preserve branch continuity
# defaults to the dry root when no previous solution is supplied
# ======================================================================

def xi_crit_liquid(p, previous_xi=None):
    # recover dry analytical solution when Omega = 0
    if abs(p.Omega) < 1e-14:
        return xi_crit_dry(p), "dry", []

    candidates = []

    # regime A no boiling
    aA, bA, cA = regime_A_coeffs(p)

    for xi in quadratic_roots(aA, bA, cA):
        if (
            xi > 1e-12
            and valid_regime_A(xi, p)
        ):
            candidates.append(("A", xi))

    # regime B partial boiling
    xiB = xi_crit_B(p)

    if (
        np.isfinite(xiB)
        and xiB > 1e-12
        and valid_regime_B(xiB, p)
    ):
        candidates.append(("B", xiB))

    # regime C complete vaporization
    aC, bC, cC = regime_C_coeffs(p)

    for xi in quadratic_roots(aC, bC, cC):
        if (
            xi > 1e-12
            and valid_regime_C(xi, p)
        ):
            candidates.append(("C", xi))

    # no physically admissible analytical threshold
    if not candidates:
        return np.nan, None, []

    # first solution follows dry root
    # later solutions follow previous branch
    if previous_xi is None or not np.isfinite(previous_xi):
        target = xi_crit_dry(p)
    else:
        target = previous_xi

    # choose admissible root nearest target
    regime, xi_best = min(
        candidates,
        key=lambda item: abs(item[1] - target),
    )

    return xi_best, regime, candidates

# ======================================================================
# vapor-only source threshold
# external water enters directly as vapor at T_w
# ======================================================================

def vapor_source_coeffs(p):
    O = p.Omega

    N0 = p.Cp_air * p.T_air

    N1 = (
        p.Cp_m * p.T_m
        + O * p.Cp_w * p.T_w
        - (1.0 + O) * p.Cp_air * p.T_air
    )

    D1 = (
        p.Cp_m
        + O * p.Cp_w
        - (1.0 + O) * p.Cp_air
    )

    R1 = (
        p.n * p.R_g
        + O * p.R_w
        - (1.0 + O) * p.R_air
    )

    a = R1 * N1

    b = (
        p.R_air * N1
        + R1 * N0
        - p.R_air * p.T_air * D1
    )

    c = 0.0

    return a, b, c


def xi_crit_vapor(
    p,
    previous_xi=None,
):
    if abs(p.Omega) < 1e-14:
        return xi_crit_dry(p)

    candidates = []

    a, b, c = vapor_source_coeffs(p)

    for xi in quadratic_roots(a, b, c):
        if xi <= 1e-12:
            continue

        if not in_bounds(xi, p):
            continue

        if not valid_air_fraction(xi, p):
            continue

        candidates.append(xi)

    if not candidates:
        return np.nan

    if previous_xi is None or not np.isfinite(previous_xi):
        target = xi_crit_dry(p)
    else:
        target = previous_xi

    return min(
        candidates,
        key=lambda xi: abs(xi - target),
    )
