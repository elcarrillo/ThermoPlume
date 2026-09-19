# interface between ThermoPlume python code and compiled C++ numerical backend
# keeps native extension details out of the public model API

from . import _thermoplume_cpp


def find_cpp_numerical_roots(
    p,
    n_scan=3000,
    xi_min=1e-8,
    root_tol=1e-12,
):
    # call compiled C++ numerical root finder
    return _thermoplume_cpp.find_numerical_roots(
        p,
        n_scan=n_scan,
        xi_min=xi_min,
        root_tol=root_tol,
    )


def find_cpp_vapor_numerical_roots(
    p,
    n_scan=3000,
    xi_min=1e-8,
    root_tol=1e-12,
):
    # call compiled C++ vapor-only numerical root finder
    return _thermoplume_cpp.find_vapor_numerical_roots(
        p,
        n_scan=n_scan,
        xi_min=xi_min,
        root_tol=root_tol,
    )
