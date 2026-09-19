import numpy as np

from thermoplume import (
    ThermoPlumeModel,
    ThermoPlumeParameters,
)


def test_vapor_model():
    previous_a = None
    previous_n = None

    for Omega in [
        0.00,
        0.05,
        0.15,
        0.30,
        0.60,
    ]:
        p = ThermoPlumeParameters(
            Omega=Omega,
            T_w=373.15,
        )

        model = ThermoPlumeModel(p)

        a = model.solve(
            method="analytical",
            water_state="vapor",
            previous_xi=previous_a,
        )

        n = model.solve(
            method="numerical",
            backend="python",
            water_state="vapor",
            previous_xi=previous_n,
        )

        print(
            f"Omega={Omega:.2f} "
            f"analytical={a['xi_crit']} "
            f"numerical={n['xi_crit']}"
        )

        if np.isnan(a["xi_crit"]):
            assert np.isnan(n["xi_crit"])

        else:
            assert np.isclose(
                a["xi_crit"],
                n["xi_crit"],
                rtol=1e-8,
                atol=1e-10,
            )

            previous_a = a["xi_crit"]
            previous_n = n["xi_crit"]


if __name__ == "__main__":
    test_vapor_model()
    print("vapor analytical and numerical roots agree")
