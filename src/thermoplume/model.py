# ======================================================================
# public interface for solving ThermoPlume neutral buoyancy thresholds
# wraps analytical and numerical solvers behind a common model API
# ======================================================================


import numpy as np

from .analytical import (
    xi_crit_dry,
    xi_crit_liquid,
)

from .numerical import (
    find_numerical_roots,
    numerical_thermo_state,
)

from .parameters import ThermoPlumeParameters


class ThermoPlumeModel:
    def __init__(self, params=None):
        # use default params when none are supplied
        if params is None:
            params = ThermoPlumeParameters()

        self.params = params

    def solve(self, method="analytical", previous_xi=None):
        # solve neutral buoyancy threshold using requested method
        if method == "analytical":
            return self._solve_analytical(previous_xi)

        if method == "numerical":
            return self._solve_numerical(previous_xi)

        raise ValueError(
            "method must be 'analytical' or 'numerical'"
        )

    def _solve_analytical(self, previous_xi):
        # closed-form analytical solution
        xi_crit, regime, _ = xi_crit_liquid(
            self.params,
            previous_xi=previous_xi,
        )

        return {
            "xi_crit": xi_crit,
            "regime": regime,
            "method": "analytical",
        }

    def _solve_numerical(self, previous_xi):
        # find all numerical neutral buoyancy roots
        roots = find_numerical_roots(
            self.params,
        )

        # no physically admissible root
        if not roots:
            return {
                "xi_crit": np.nan,
                "regime": None,
                "method": "numerical",
            }

        # first solution follows dry analytical root
        # later solutions follow previous branch
        if previous_xi is None or not np.isfinite(previous_xi):
            target = xi_crit_dry(
                self.params,
            )
        else:
            target = previous_xi

        # select numerical root nearest target branch
        xi_crit = min(
            roots,
            key=lambda xi: abs(xi - target),
        )

        # identify thermodynamic regime at selected root
        if abs(self.params.Omega) < 1e-14:
            regime = "dry"
        else:
            regime = numerical_thermo_state(
                xi_crit,
                self.params,
            )["regime"]

        return {
            "xi_crit": xi_crit,
            "regime": regime,
            "method": "numerical",
        }