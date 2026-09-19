#pragma once


namespace thermoplume {

struct Parameters {
    // source and water loading
    double Omega = 0.05;
    double n = 0.03;

    // temp K
    double T_m = 1100.0;
    double T_air = 273.15;
    double T_w = 373.15;
    double T_w0 = 273.15;
    double T_sat = 373.15;

    // heat capacity J kg^-1 K^-1
    double Cp_air = 1004.0;
    double Cp_m = 3000.0;
    double Cp_w = 1850.0;
    double Cp_l = 4180.0;

    // gas constant J kg^-1 K^-1
    double R_air = 287.0;
    double R_g = 461.5;
    double R_w = 461.5;

    // latent heat J kg^-1
    double L_v = 2.5e6;

    // plume and density params
    double C = 2.0;
    double k = 0.05;
    double rho_air = 1.2;
    double rho_a = 1.2;

    // gravity and source geometry
    double g = 9.81;
    double u0 = 100.0;
    double D = 50.0;

    // numerical floor
    double Y_air_min = 1e-10;
};

}  // namespace thermoplume