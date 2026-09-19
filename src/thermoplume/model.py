# =====================================================================
# public interface for solving ThermoPlume neutral buoyancy thresholds
# wraps analytical and numerical solvers behind a common model API
# =====================================================================

import numpy as np

from .analytical import (
    xi_crit_dry,
    xi_crit_liquid,
    xi_crit_vapor,
)

from .cpp_backend import (
    find_cpp_numerical_roots,
    find_cpp_vapor_numerical_roots,
)

from .numerical import (
    find_numerical_roots,
    find_vapor_numerical_roots,
    numerical_thermo_state,
)

from .parameters import ThermoPlumeParameters


class ThermoPlumeModel:
    def __init__(self, params=None):
        # use default params when none are supplied
        if params is None:
            params = ThermoPlumeParameters()

        self.params = params

    def solve(
        self,
        method="analytical",
        backend="python",
        previous_xi=None,
        water_state="liquid",
    ):
        if water_state not in {
            "liquid",
            "vapor",
        }:
            raise ValueError(
                "water_state must be 'liquid' or 'vapor'"
            )

        if method == "analytical":
            return self._solve_analytical(
                previous_xi,
                water_state,
            )

        if method == "numerical":
            return self._solve_numerical(
                backend,
                previous_xi,
                water_state,
            )

        raise ValueError(
            "method must be 'analytical' or 'numerical'"
        )

    def _solve_analytical(
        self,
        previous_xi,
        water_state,
    ):
        if water_state == "liquid":
            xi_crit, regime, _ = xi_crit_liquid(
                self.params,
                previous_xi=previous_xi,
            )

        else:
            xi_crit = xi_crit_vapor(
                self.params,
                previous_xi=previous_xi,
            )

            if abs(self.params.Omega) < 1e-14:
                regime = "dry"

            elif np.isfinite(xi_crit):
                regime = "vapor"

            else:
                regime = None

        return {
            "xi_crit": xi_crit,
            "regime": regime,
            "method": "analytical",
            "backend": "python",
        }

    def _solve_numerical(
        self,
        backend,
        previous_xi,
        water_state,
    ):
        if water_state == "liquid":
            if backend == "python":
                roots = find_numerical_roots(
                    self.params,
                )

            elif backend == "cpp":
                roots = find_cpp_numerical_roots(
                    self.params,
                )

            else:
                raise ValueError(
                    "backend must be 'python' or 'cpp'"
                )

        else:
            if backend == "python":
                roots = find_vapor_numerical_roots(
                    self.params,
                )

            elif backend == "cpp":
                roots = find_cpp_vapor_numerical_roots(
                    self.params,
                )

            else:
                raise ValueError(
                    "backend must be 'python' or 'cpp'"
                )

        if not roots:
            return {
                "xi_crit": np.nan,
                "regime": None,
                "method": "numerical",
                "backend": backend,
            }

        if (
            previous_xi is None
            or not np.isfinite(previous_xi)
        ):
            target = xi_crit_dry(
                self.params,
            )

        else:
            target = previous_xi

        xi_crit = min(
            roots,
            key=lambda xi: abs(xi - target),
        )

        if abs(self.params.Omega) < 1e-14:
            regime = "dry"

        elif water_state == "vapor":
            regime = "vapor"

        else:
            regime = numerical_thermo_state(
                xi_crit,
                self.params,
            )["regime"]

        return {
            "xi_crit": xi_crit,
            "regime": regime,
            "method": "numerical",
            "backend": backend,
        }
