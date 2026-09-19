#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "thermoplume/numerical.hpp"
#include "thermoplume/parameters.hpp"


namespace py = pybind11;


namespace {

thermoplume::Parameters params_from_python(
    const py::object& params
) {
    thermoplume::Parameters p;

    p.Omega = params.attr("Omega").cast<double>();
    p.n = params.attr("n").cast<double>();

    p.T_m = params.attr("T_m").cast<double>();
    p.T_air = params.attr("T_air").cast<double>();
    p.T_w = params.attr("T_w").cast<double>();
    p.T_w0 = params.attr("T_w0").cast<double>();
    p.T_sat = params.attr("T_sat").cast<double>();

    p.Cp_air = params.attr("Cp_air").cast<double>();
    p.Cp_m = params.attr("Cp_m").cast<double>();
    p.Cp_w = params.attr("Cp_w").cast<double>();
    p.Cp_l = params.attr("Cp_l").cast<double>();

    p.R_air = params.attr("R_air").cast<double>();
    p.R_g = params.attr("R_g").cast<double>();
    p.R_w = params.attr("R_w").cast<double>();

    p.L_v = params.attr("L_v").cast<double>();

    p.C = params.attr("C").cast<double>();
    p.k = params.attr("k").cast<double>();
    p.rho_air = params.attr("rho_air").cast<double>();
    p.rho_a = params.attr("rho_a").cast<double>();

    p.g = params.attr("g").cast<double>();
    p.u0 = params.attr("u0").cast<double>();
    p.D = params.attr("D").cast<double>();

    p.Y_air_min =
        params.attr("Y_air_min").cast<double>();

    return p;
}

}  // namespace


PYBIND11_MODULE(_thermoplume_cpp, m) {
    m.doc() =
        "C++ numerical backend for ThermoPlume";

    m.def(
        "find_numerical_roots",
        [](const py::object& params,
           int n_scan,
           double xi_min,
           double root_tol) {
            const thermoplume::Parameters p =
                params_from_python(params);

            return thermoplume::find_numerical_roots(
                p,
                n_scan,
                xi_min,
                root_tol
            );
        },
        py::arg("params"),
        py::arg("n_scan") = 3000,
        py::arg("xi_min") = 1e-8,
        py::arg("root_tol") = 1e-12
    );


    m.def(
        "find_vapor_numerical_roots",
        [](const py::object& params,
           int n_scan,
           double xi_min,
           double root_tol) {
            const thermoplume::Parameters p =
                params_from_python(params);

            return thermoplume::find_vapor_numerical_roots(
                p,
                n_scan,
                xi_min,
                root_tol
            );
        },
        py::arg("params"),
        py::arg("n_scan") = 3000,
        py::arg("xi_min") = 1e-8,
        py::arg("root_tol") = 1e-12
    );
}
