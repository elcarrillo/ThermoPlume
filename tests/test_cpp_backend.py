# checks compiled C++ numerical backend against Python numerical solver
# both implementations should return matching neutral buoyancy thresholds


import numpy as np

from thermoplume import (
    ThermoPlumeModel,
    ThermoPlumeParameters,
)


def test_cpp_backend():
    Omega_values = [
        0.0,
        0.05,
        0.15,
        0.30,
        0.60,
    ]

    for Omega in Omega_values:
        params = ThermoPlumeParameters(
            Omega=Omega,
        )

        model = ThermoPlumeModel(
            params,
        )

        python_result = model.solve(
            method="numerical",
            backend="python",
        )

        cpp_result = model.solve(
            method="numerical",
            backend="cpp",
        )

        print(
            f"Omega={Omega:.2f} "
            f"python={python_result['xi_crit']} "
            f"cpp={cpp_result['xi_crit']}"
        )

        assert (
            python_result["regime"]
            == cpp_result["regime"]
        )

        if np.isnan(
            python_result["xi_crit"]
        ):
            assert np.isnan(
                cpp_result["xi_crit"]
            )

        else:
            assert np.isclose(
                python_result["xi_crit"],
                cpp_result["xi_crit"],
                rtol=1e-8,
                atol=1e-10,
            )


if __name__ == "__main__":
    test_cpp_backend()

    print("Python and C++ numerical backends agree")