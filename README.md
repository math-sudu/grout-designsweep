# Cement-time frontiers for tunnel support

Reproduction code for *Cement-time frontiers for tunnel support using
hydration-dependent stiffness*. Three white-cement pastes (W32/W40/W48) supply
the paired calorimetry and modulus benchmark. The code constructs loading
events, minimum-cement frontiers, adjacent time costs and reaction requirements
for a common section-compliance demand.

## Environment

Use Python 3.10 or newer for the material model, postprocessing and plots, and
Docker with Linux containers for the finite-element calculations. Run every
command from this repository root.

```text
python -m pip install -e ".[test]"
docker build -t grout-support-fe .
```

The Dockerfile pins the FE image by digest and adds PyYAML. It contains Python
3.12.3, DOLFINx 0.11.0.post0, Gmsh 4.15.2, NumPy 2.4.6 and SciPy 1.17.1.
The host workflow was run with Python 3.13, NumPy 2.4.4, SciPy 1.16.2, pandas 2.3.3,
Matplotlib 3.10.6 and PyMuPDF 1.27.2.2. Figure export requires an installed
Times New Roman font; STIX math fonts come with Matplotlib. The font files are
not included. The bundled Origin-style exporter and its JSON configuration
come from GeoSkills commit `c274d819`; they have no dependency on that checkout.

## Reproduce the study

1. Download the public accepted manuscript and extract its vector curves:

   ```text
   python scripts/prepare_paste_benchmark.py --download
   ```

   This obtains [Dunant et al. (2020)](https://doi.org/10.1016/j.cemconres.2020.106007)
   from [Cambridge Apollo](https://doi.org/10.17863/CAM.50607). Figures 5 and 10
   produce `data/public_paste_benchmark/heat_flow.csv` and `modulus.csv`.
   These are figure-derived inputs, with extraction details in `metadata.json`.
   The third-party PDF stays under its repository license and is downloaded
   separately. `thermal_basis.json` records the physical input sources and
   transfer assumptions. Section dimensions follow
   [Liu et al. (2021)](https://doi.org/10.1016/j.tust.2021.103857).

2. Rebuild the uniform-annulus FE table directly from `configs/support_design.yaml`:

   ```powershell
   pwsh -NoProfile -File scripts/run_in_dolfinx.ps1 scripts/prepare_support_response.py
   ```

   On a POSIX shell, the equivalent container invocation is:

   ```sh
   docker run --rm -v "$PWD:/work" -w /work grout-support-fe bash -lc \
     'export PYTHONPATH="/work/src:$PYTHONPATH"; python3 scripts/prepare_support_response.py'
   ```

   Substitute the Python script and arguments in that invocation for subsequent
   FE steps. Both the image and PowerShell entry set `OMP_NUM_THREADS=1` and
   `OPENBLAS_NUM_THREADS=1`. This step produces the 143-point
   `results/support_design/reference_response.npz`, including 20 independent
   check points. No previous FE arrays or fitted pressure records are required;
   compliance is normalized by the specified linear traction.

3. Generate thermal query inputs and solve the radial stiffness fields:

   ```text
   python scripts/run_support_queries.py
   pwsh -NoProfile -File scripts/run_in_dolfinx.ps1 scripts/run_radial_support.py --workers 8
   ```

   Choose a worker count that fits available CPU and memory. The main design
   uses 363 common-origin trajectories (11 temperature cases, three pastes,
   11 thermal parameter scenarios). `query_runs.csv` also retains the separate
   3 h/6 h reference-state cases; radial selection excludes those cases.
   `radial_histories.csv`, `radial_attainment.csv` and `radial_runs.csv` contain
   the field responses, 41-demand events and inputs. Add `--resume` to continue
   an interrupted radial run. A selected run such as `--case reference
   --scenario nominal` overwrites these files unless `--resume` is supplied;
   use the complete run for the full design tables.

4. Reconstruct the paper's design outputs and four figures:

   ```text
   python scripts/run_support_events.py
   python scripts/run_support_reaction.py
   python scripts/run_support_updates.py
   python scripts/plot_support_paper.py
   ```

   `event_designs.csv` contains 55,022 grid decisions;
   `cement_time_frontier.csv` contains 2,610 intervals with explicit endpoint
   inclusion; `cement_time_tradeoffs.csv` contains 985 adjacent transitions.
   `uniform_reaction_requirements.csv` and `attainment_reaction.csv` report
   heat requirements; `state_updates.csv` reports fixed-material clock updates.
   Figures are written as PDF and editable SVG under `paper/figures/`.
   Generated outputs are recreated locally and excluded from Git.

For the reference 2.2 micrometres/kPa demand, nominal radial attainment is
1.642586/2.396481/3.590246 h for W32/W40/W48. The adjacent cement savings are
174.9/139.8 kg per initial cubic metre for 0.754/1.194 h additional time.
With paste at 16 degrees C and lining/soil at 30 degrees C, all-scenario
selection at 4 h is W40; W48 becomes available at 4.005554 h.

## Checks and additional calculations

```text
python -m pytest tests -q
pwsh -NoProfile -File scripts/run_in_dolfinx.ps1 scripts/check_support_profiles.py --critical-refinement
```

Run the tests after generating the curve CSVs and response table. They cover
the reaction-coordinate inverse, initial-state continuation, event feasibility,
radial crossing interpolation and frontier endpoints. The critical refinement
checks the cold-paste/warm-surroundings W48 decision at 4 h on a finer mesh.

`python scripts/run_paste_support.py` recreates the representative temperature
histories used during method development. `check_support_profiles.py` uses
those outputs for its default checks, or the query outputs with
`--initial-state-queries` and `--design-decisions`. These diagnostics are separate
from the main figure-generation sequence above.
