# checks analytical model against numerical method
# analytical root should match one numerical root at the same model params


from dataclasses import replace

import numpy as np

from thermoplume.analytical import xi_crit_liquid
from thermoplume.numerical import find_numerical_roots
from thermoplume.parameters import ThermoPlumeParameters


def test_solver_agreement():
    base = ThermoPlumeParameters()

    Omega_values = [
        0.0,
        0.05,
        0.15,
        0.30,
        0.60,
    ]

    previous_xi = None

    for Omega in Omega_values:
        p = replace(
            base,
            Omega=Omega,
        )

        # analytical root
        xi_analytical, regime, _ = xi_crit_liquid(
            p,
            previous_xi=previous_xi,
        )

        # all numerical roots
        numerical_roots = find_numerical_roots(p)

        print(
            f"Omega={Omega:.2f} "
            f"regime={regime} "
            f"analytical={xi_analytical} "
            f"numerical={numerical_roots}"
        )

        # analytical solver found no physical root
        if not np.isfinite(xi_analytical):
            assert not numerical_roots
            continue

        # find numerical root nearest analytical solution
        xi_numerical = min(
            numerical_roots,
            key=lambda xi: abs(xi - xi_analytical),
        )

        # analytical and numerical roots should agree
        assert np.isclose(
            xi_analytical,
            xi_numerical,
            rtol=1e-8,
            atol=1e-10,
        ), (
            f"root mismatch "
            f"Omega={Omega} "
            f"analytical={xi_analytical} "
            f"numerical={xi_numerical}"
        )

        previous_xi = xi_analytical


if __name__ == "__main__":
    test_solver_agreement()

    print("analytical and numerical roots agree")