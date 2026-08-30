# ThermoPlume

ThermoPlume is a Python package for calculating thermodynamic neutral-buoyancy thresholds in volcanic eruption plumes, including the effects of external water.

It includes analytical solutions and an independent numerical solver for thermodynamic regimes with no boiling, partial boiling, and complete vaporization.

## Installation

```bash
git clone git@github.com:elcarrillo/thermoplume.git
cd thermoplume
pip install -e .
````

## Quick start

```python
from thermoplume import (
    ThermoPlumeModel,
    ThermoPlumeParameters,
)

params = ThermoPlumeParameters(
    Omega=0.15,
    T_m=1100.0,
)

model = ThermoPlumeModel(params)

result = model.solve(
    method="analytical",
)

print(result)
```

For the numerical solver:

```python
result = model.solve(
    method="numerical",
)
```

## Tests

```bash
python tests/test_solver_agreement.py
python tests/test_model.py
```

## License

See [LICENSE](LICENSE).

