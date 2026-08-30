# computes source-loading values where the thermodynamic state changes regime
# xi_AB marks the onset of boiling when available enthalpy reaches saturation
# xi_BC marks complete vaporization when available enthalpy meets latent heat demand
# used by the numerical solver to place exact regime transitions in the root scan


def regime_boundaries(p):
    # H_i = H0 + H1 xi
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

    xi_AB = None
    xi_BC = None

    # onset of boiling H_i = 0
    if abs(H1) > 1e-14:
        xi_AB = -H0 / H1

    # complete vaporization H_i = omega L_v
    denom_BC = H1 - p.Omega * p.L_v

    if abs(denom_BC) > 1e-14:
        xi_BC = -H0 / denom_BC

    return xi_AB, xi_BC