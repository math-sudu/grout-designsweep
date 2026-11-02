# Grout selection under staged loading

Code and digitized input data for **Grout selection for shield tunnels under
staged loading using physics-informed neural networks**.

The calculation combines measured paste hardening, physics-informed incremental
displacement and stress responses, and linear programming. It selects a material
whose loading schedule satisfies intermediate pressure requirements, the final
pressure and surface-displacement bounds. The supplied W32, W40 and W48 inputs
are paired calorimetry and Young's modulus curves from
[Dunant et al. (2020)](https://doi.org/10.1016/j.cemconres.2020.106007).

## Installation

Use Python 3.10 or newer and run commands from this directory:

```text
python -m pip install -e ".[test]"
```

The calculations used Python 3.13.7, PyTorch 2.8.0, NumPy 2.4.4, SciPy 1.16.2,
pandas 2.3.3 and PyMuPDF 1.27.2.2. Thermal training can use CPU or CUDA;
mechanical least-squares calculations use CPU.

## Inputs

`data/public_paste_benchmark/heat_flow.csv` and `modulus.csv` contain the curves
digitized from Figures 5 and 10 of the accepted manuscript. The accompanying
metadata records their units and extraction method. The original article is
available through [Cambridge Apollo](https://doi.org/10.17863/CAM.50607).
To repeat the extraction:

```text
python scripts/prepare_paste_benchmark.py --download
```

`configs/support_design.yaml` supplies the geometry and material constants.
`configs/support_handover.yaml` specifies loading times and displacement bounds.
`configs/pinn_support.json` and the material-specific `configs/W32_event.json`,
`W40_event.json` and `W48_event.json` contain neural and numerical settings.

Elapsed assessment time begins at the common measured reference age of
1.68637216 h. The prescribed four-hour loading window starts from that state.
Mechanical activation occurs at the first pressure withdrawal. Further loading
retains that activation reference and the displacement and stress already
accumulated.

## Reproduce the calculations

Train and evaluate each material, replacing `W32` with `W40` and `W48`:

```text
python scripts/train_pinn_thermal.py --mix W32 --device cpu
python scripts/check_pinn_thermal.py --directory results/pinn_support/W32/thermal_age
python scripts/run_pinn_event.py --mix W32
python scripts/run_pinn_pressure.py --mix W32
python scripts/run_pinn_design.py --mix W32 --continuous
```

Once all three materials are available, construct the initially loaded W40
state and evaluate the loading schedules:

```text
python scripts/check_pinn_history.py --continuous-pressure
python scripts/run_pinn_design.py --qualification
python scripts/run_pinn_plans.py
python scripts/check_pinn_decisions.py
```

These commands create models and calculation records under `results/pinn_support/`.
The last command computes the independent, richer incremental responses used
to assess the material choices and further loading.

For a final pressure of 100 kPa and withdrawal-induced settlement limited to
0.220 mm, the cumulative fractions 0.2/0.5/1 at 0.37/1.73/4 h select W32.
Moving the intermediate times to 1.73/3.1 h selects W40. At equal initial fill
volume, W40 contains 174.9 kg/m3 less cement. Further loading of W40 retains
the 20% pressure applied at 0.37 h and applies the remaining pressure within
the original displacement bounds.

Run the supplied tests with:

```text
python -m pytest tests -q
```

## Method references

- [Liu et al. (2021)](https://doi.org/10.1016/j.tust.2021.103857): section
  geometry and pressure-support withdrawal sequence.
- [Dong and Li (2021)](https://doi.org/10.1016/j.cma.2021.114129): fixed neural
  features and least-squares training.
- [Harandi et al. (2024)](https://doi.org/10.1002/nme.7388): mixed field approximation.
- [Galitz and Grasley (2016)](https://doi.org/10.1617/s11527-015-0576-8):
  solidification history and its nonrelaxing elastic limit.

Each material's spatial features are combined with a cubic time basis, with
the product coefficients trained jointly along its measured hardening path.
Evaluating the trained coefficients at a proposed application time supplies a
new incremental response. The solidification law retains previous responses,
and linear programming allocates the pressure still to be applied.
