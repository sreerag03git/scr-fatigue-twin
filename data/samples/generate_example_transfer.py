"""Generate an EXAMPLE imported-H(f) CSV (the format the 'imported' route consumes).

This is an ILLUSTRATIVE table (sampled from the reference wave-band shape), NOT a
measured OrcaFlex/RIFLEX/DeepLines result. It documents the CSV format so a project
can drop in its own *validated* complex transfer function with the same columns:

    freq_hz, magnitude [N.m per m heave], phase_rad

(real/imag columns `freq, re, im` are also accepted; phase in degrees if the column
is named `phase_deg`). Provenance is read from the `# key: value` header lines.

Run:  python data/samples/generate_example_transfer.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "core"))
from scr_twin_core.transfer import reference_transfer_function  # noqa: E402

OUT = pathlib.Path(__file__).with_name("example_transfer_function.csv")


def main() -> None:
    f = np.linspace(0.02, 0.45, 60)
    tf = reference_transfer_function(f)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        fh.write("# EXAMPLE imported H(f) - ILLUSTRATIVE format template, NOT measured data.\n")
        fh.write("# source_tool: EXAMPLE (replace with your OrcaFlex / RIFLEX / DeepLines export)\n")
        fh.write("# tool_version: n/a\n")
        fh.write("# load_case: reference SCR, moderate sea state (illustrative)\n")
        fh.write("# columns: freq_hz, magnitude [N.m per m heave], phase_rad\n")
        fh.write("freq_hz,magnitude,phase_rad\n")
        for fi, mi, pi in zip(f, tf.magnitude, tf.phase, strict=False):
            fh.write(f"{fi:.4f},{mi:.6e},{pi:.6f}\n")
    print(f"wrote {OUT} ({f.size} rows)")


if __name__ == "__main__":
    main()
