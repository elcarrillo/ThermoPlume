import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermoplume import (
    ThermoPlumeModel,
    ThermoPlumeParameters,
)


# ------------------------------------------------------------
# output
# ------------------------------------------------------------

outdir = Path("notebooks/data/validation")
outdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "analytical_numerical_validation.csv"
png_path = outdir / "analytical_numerical_validation.png"


# ------------------------------------------------------------
# validation parameter space
#
# use broad temperature range so all three liquid-water
# thermodynamic regimes are sampled
# ------------------------------------------------------------

temperatures_C = np.arange(
    200.0,
    1200.0 + 1.0,
    50.0,
)

omegas = np.linspace(
    0.0,
    0.60,
    121,
)


# ------------------------------------------------------------
# support both current and earlier solve APIs
# ------------------------------------------------------------

solve_parameters = inspect.signature(
    ThermoPlumeModel.solve
).parameters

supports_water_state = (
    "water_state" in solve_parameters
)


def solve_model(
    model,
    method,
    previous_xi,
    backend=None,
):
    kwargs = {
        "method": method,
        "previous_xi": previous_xi,
    }

    if backend is not None:
        kwargs["backend"] = backend

    if supports_water_state:
        kwargs["water_state"] = "liquid"

    return model.solve(**kwargs)


def normalize_regime(regime):
    if regime is None:
        return None

    text = str(regime).strip().upper()

    if text == "DRY":
        return "dry"

    for name in ("A", "B", "C"):
        if (
            text == name
            or text.endswith(name)
            or f"REGIME {name}" in text
        ):
            return name

    return str(regime)


# ------------------------------------------------------------
# run analytical and numerical models
# ------------------------------------------------------------

rows = []

for T_m_C in temperatures_C:

    previous_analytical = None
    previous_numerical = None

    for Omega in omegas:

        params = ThermoPlumeParameters(
            Omega=float(Omega),
            n=0.03,
            T_m=float(T_m_C + 273.15),
            T_air=273.15,
            T_w0=273.15,
            T_sat=373.15,
            Cp_m=3000.0,
        )

        model = ThermoPlumeModel(params)

        analytical = solve_model(
            model=model,
            method="analytical",
            previous_xi=previous_analytical,
        )

        numerical = solve_model(
            model=model,
            method="numerical",
            backend="python",
            previous_xi=previous_numerical,
        )

        xi_a = float(
            analytical["xi_crit"]
        )

        xi_n = float(
            numerical["xi_crit"]
        )

        finite_a = np.isfinite(xi_a)
        finite_n = np.isfinite(xi_n)

        if finite_a:
            previous_analytical = xi_a
        else:
            previous_analytical = np.nan

        if finite_n:
            previous_numerical = xi_n
        else:
            previous_numerical = np.nan

        regime_a = normalize_regime(
            analytical["regime"]
        )

        regime_n = normalize_regime(
            numerical["regime"]
        )

        if finite_a and finite_n:
            abs_error = abs(
                xi_n - xi_a
            )

            rel_error = (
                abs_error / abs(xi_a)
                if xi_a != 0.0
                else np.nan
            )

        else:
            abs_error = np.nan
            rel_error = np.nan

        rows.append(
            {
                "T_m_C": T_m_C,
                "Omega": Omega,
                "xi_analytical": xi_a,
                "xi_numerical": xi_n,
                "regime_analytical": regime_a,
                "regime_numerical": regime_n,
                "root_exists_analytical": finite_a,
                "root_exists_numerical": finite_n,
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
        )


df = pd.DataFrame(rows)

df.to_csv(
    csv_path,
    index=False,
)


# ------------------------------------------------------------
# verification checks
# ------------------------------------------------------------

root_mismatch = (
    df["root_exists_analytical"]
    != df["root_exists_numerical"]
)

if root_mismatch.any():
    bad = df.loc[
        root_mismatch,
        [
            "T_m_C",
            "Omega",
            "xi_analytical",
            "xi_numerical",
        ],
    ]

    raise RuntimeError(
        "analytical/numerical root-existence mismatch:\n"
        + bad.to_string(index=False)
    )


finite = (
    df["root_exists_analytical"]
    & df["root_exists_numerical"]
)

wet = (
    finite
    & (df["Omega"] > 0.0)
)

regime_mismatch = (
    wet
    & (
        df["regime_analytical"]
        != df["regime_numerical"]
    )
)

if regime_mismatch.any():
    bad = df.loc[
        regime_mismatch,
        [
            "T_m_C",
            "Omega",
            "regime_analytical",
            "regime_numerical",
        ],
    ]

    raise RuntimeError(
        "analytical/numerical regime mismatch:\n"
        + bad.to_string(index=False)
    )


comparison = df.loc[
    finite
].copy()

max_abs_error = (
    comparison["abs_error"].max()
)

max_rel_error = (
    comparison["rel_error"].max()
)


# ------------------------------------------------------------
# figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.6),
)

regime_markers = {
    "A": "o",
    "B": "s",
    "C": "^",
}


# ------------------------------------------------------------
# analytical versus numerical
# ------------------------------------------------------------

ax = axes[0]

for regime in ("A", "B", "C"):

    subset = comparison.loc[
        comparison["regime_analytical"]
        == regime
    ]

    if subset.empty:
        continue

    ax.scatter(
        subset["xi_analytical"],
        subset["xi_numerical"],
        marker=regime_markers[regime],
        s=16,
        alpha=0.45,
        label=f"Regime {regime}",
    )

limits = [
    min(
        comparison["xi_analytical"].min(),
        comparison["xi_numerical"].min(),
    ),
    max(
        comparison["xi_analytical"].max(),
        comparison["xi_numerical"].max(),
    ),
]

ax.plot(
    limits,
    limits,
    linestyle="--",
    linewidth=1.4,
    label="1:1",
)

ax.set_xlim(limits)
ax.set_ylim(limits)

ax.set_xlabel(
    r"Analytical $\xi_{\rm crit}$"
)

ax.set_ylabel(
    r"Numerical $\xi_{\rm crit}$"
)

ax.set_title(
    "(a) Analytical–numerical agreement"
)

ax.legend(
    frameon=False
)


# ------------------------------------------------------------
# numerical error
# ------------------------------------------------------------

ax = axes[1]

error_floor = np.finfo(float).eps

for regime in ("A", "B", "C"):

    subset = comparison.loc[
        comparison["regime_analytical"]
        == regime
    ]

    if subset.empty:
        continue

    ax.scatter(
        subset["Omega"],
        np.maximum(
            subset["abs_error"],
            error_floor,
        ),
        marker=regime_markers[regime],
        s=16,
        alpha=0.45,
    )

ax.axhline(
    error_floor,
    linestyle="--",
    linewidth=1.2,
    label="machine precision",
)

ax.set_yscale("log")

ax.set_xlabel(
    r"External-water loading $\Omega$"
)

ax.set_ylabel(
    r"$|\xi_{\rm num}-\xi_{\rm analytic}|$"
)

ax.set_title(
    "(b) Numerical error"
)

ax.legend(
    frameon=False
)


fig.tight_layout()

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# report
# ------------------------------------------------------------

print(
    f"finite analytical/numerical comparisons: "
    f"{len(comparison)}"
)

for regime in ("A", "B", "C"):
    count = (
        comparison["regime_analytical"]
        == regime
    ).sum()

    print(
        f"Regime {regime}: {count}"
    )

print(
    f"max absolute error: "
    f"{max_abs_error:.6e}"
)

print(
    f"max relative error: "
    f"{max_rel_error:.6e}"
)

print(
    f"saved {csv_path}"
)

print(
    f"saved {png_path}"
)

