from dataclasses import dataclass


@dataclass
class ThermoPlumeParameters:
    # source and water loading
    Omega: float = 0.05
    n: float = 0.03

    # temp K
    T_m: float = 1100.0
    T_air: float = 273.15
    T_w0: float = 273.15
    T_sat: float = 373.15

    # heat capacity J kg^-1 K^-1
    Cp_air: float = 1004.0
    Cp_m: float = 3000.0
    Cp_w: float = 1850.0
    Cp_l: float = 4180.0

    # gas constant J kg^-1 K^-1
    R_air: float = 287.0
    R_g: float = 461.5
    R_w: float = 461.5

    # latent heat J kg^-1
    L_v: float = 2.5e6

    # plume and density params
    C: float = 2.0
    k: float = 0.05
    rho_air: float = 1.2
    rho_a: float = 1.2

    # gravity and source geometry
    g: float = 9.81
    u0: float = 100.0
    D: float = 50.0

    # numerical floor
    Y_air_min: float = 1e-10
