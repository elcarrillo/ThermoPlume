## =============================================
## checks user/public thermoplume model interface 
## =============================================

import numpy as np

from thermoplume import (
    ThermoPlumeModel,
    ThermoPlumeParameters,
)


def test_model_api():
    params = ThermoPlumeParameters(
        Omega=0.15,
        T_m=1100.0,
    )

    model = ThermoPlumeModel(params)

    analytical = model.solve(
        method="analytical",
    )

    numerical = model.solve(
        method="numerical",
    )

    # expected result fields
    assert analytical["method"] == "analytical"
    assert numerical["method"] == "numerical"

    assert analytical["regime"] == numerical["regime"]

    # analytical and numerical thresholds should agree
    assert np.isclose(
        analytical["xi_crit"],
        numerical["xi_crit"],
        rtol=1e-8,
        atol=1e-10,
    )


if __name__ == "__main__":
    test_model_api()

    print("thermoplume model api works")
