"""Build the uniform-annulus response table from the configured FE section.

Run inside the image built from Dockerfile, using scripts/run_in_dolfinx.ps1.
The table requires no previous FE result or historical geometry array.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grouting_pino.support_response import SectionResponse
from support_section import RadialSection


def prepare(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    radial = RadialSection(config_path)
    low = np.geomspace(1e6, 375e6, 96)
    high = np.geomspace(375e6, 20e9, 48)
    modulus = np.r_[low, high[1:]]
    indices = np.linspace(0, len(low)-2, 12, dtype=int)
    check_modulus = np.r_[np.sqrt(low[indices] * low[indices+1]),
                         np.sqrt(high[:-1:6] * high[1::6])]
    responses = []
    for i, value in enumerate(np.r_[modulus, check_modulus], 1):
        responses.append(radial.response([0.0], [value]) * 1e-9)
        if i % 20 == 0 or i == len(modulus) + len(check_modulus):
            print(f"Solved {i}/{len(modulus)+len(check_modulus)} uniform states.", flush=True)
    section = SectionResponse(modulus, np.array(responses[:len(modulus)]), radial.geometry)
    check_compliance = np.array(responses[len(modulus):])
    output = ROOT / config["response"]["table"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, modulus_pa=modulus,
                        compliance_m_per_pa=section.compliance_m_per_pa,
                        check_modulus_pa=check_modulus,
                        check_compliance_m_per_pa=check_compliance,
                        geometry_json=np.array(json.dumps(section.geometry)))
    return output, section, check_modulus, check_compliance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/support_design.yaml")
    args = parser.parse_args()
    output, section, check_modulus, check_compliance = prepare(args.config)
    response_error = np.abs(section.compliance(check_modulus) / check_compliance - 1)
    inverted = np.array([section.required_modulus_pa(float(c)) for c in check_compliance])
    print(f"Response table: {output.relative_to(ROOT)}")
    print(f"Off-grid FE checks: {len(check_modulus)}; maximum response interpolation error "
          f"{response_error.max():.4%}; maximum modulus inversion error "
          f"{np.max(np.abs(inverted / check_modulus - 1)):.4%}")


if __name__ == "__main__":
    main()
