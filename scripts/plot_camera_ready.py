#!/usr/bin/env python3
"""Generate the camera-ready figure set from frozen result artifacts."""


from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

PRODUCERS = {
    "combined": ROOT / "scripts/_plotting/main_and_animation.py",
    "three": ROOT / "scripts/_plotting/paired_diagnostics.py",
    "v3": ROOT / "scripts/_plotting/eicu.py",
    "safety": ROOT / "scripts/_plotting/safety.py",
}


FINAL_ARTIFACTS = {
    "main/fig2_icu_fixed_schedule_comparison.png":
        ("combined", "main/fig2_icu_fixed_schedule_comparison.png"),

    "main/fig3_paired_training_vs_posthoc.png":
        ("three", "main/fig3_paired_training_vs_posthoc.png"),

    "main/fig4_full_component_ablation.png":
        ("combined", "main/fig4_full_component_ablation.png"),

    "appendix/figA1_cross_source_portability.png":
        ("three", "appendix/figA1_cross_source_portability.png"),

    "appendix/figA2_eicu_fixed_schedule_comparison.png":
        ("v3", "appendix/figA2_eicu_fixed_schedule_comparison.png"),

    "appendix/figB1_policy_distribution_diagnostics.png":
        ("three", "appendix/figB1_policy_distribution_diagnostics.png"),

    "appendix/figC1_safety_failure_diagnostics.png":
        ("safety", "appendix/figC1_safety_failure_diagnostics.png"),

    "supplementary/trajectory_animation.gif":
        ("combined", "supplementary/trajectory_animation.gif"),
}


def run(command):
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
    )


def validate_outputs(out_dir: Path):
    for relative in FINAL_ARTIFACTS:
        path = out_dir / relative

        if not path.is_file():
            raise RuntimeError(
                f"Missing final artifact: {path}"
            )

        with Image.open(path) as image:
            if path.suffix.lower() == ".png":
                dpi = image.info.get("dpi")

                if dpi is None:
                    raise RuntimeError(
                        f"Missing DPI metadata: {path}"
                    )

                if not (
                    590 <= dpi[0] <= 610
                    and 590 <= dpi[1] <= 610
                ):
                    raise RuntimeError(
                        f"Unexpected DPI {dpi}: {path}"
                    )

            elif path.suffix.lower() == ".gif":
                if getattr(image, "n_frames", 1) <= 1:
                    raise RuntimeError(
                        f"GIF is not animated: {path}"
                    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/fixed_schedule_2026"),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/camera_ready"),
    )

    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()

    for name, path in PRODUCERS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen plotting producer "
                f"{name}: {path}"
            )

    with tempfile.TemporaryDirectory(
        prefix="laadan-camera-ready-"
    ) as temporary:
        temporary = Path(temporary)

        combined = temporary / "combined"
        three = temporary / "three"
        v3 = temporary / "v3"
        safety = temporary / "safety"

        combined.mkdir()
        three.mkdir()
        v3.mkdir()
        safety.mkdir()

        run([
            sys.executable,
            str(PRODUCERS["combined"]),
            "--results-root",
            str(results_root),
            "--out-dir",
            str(combined),
            "--animate",
        ])

        run([
            sys.executable,
            str(PRODUCERS["three"]),
            "--results-root",
            str(results_root),
            "--out-dir",
            str(three),
        ])

        run([
            sys.executable,
            str(PRODUCERS["v3"]),
            "--results-root",
            str(results_root),
            "--outdir",
            str(v3),
        ])

        safety_output = (
            safety
            / "appendix"
            / "figC1_safety_failure_diagnostics.png"
        )

        safety_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        run([
            sys.executable,
            str(PRODUCERS["safety"]),
            "--results-root",
            str(results_root),
            "--output",
            str(safety_output),
        ])

        source_roots = {
            "combined": combined,
            "three": three,
            "v3": v3,
            "safety": safety,
        }

        for destination_rel, (
            source_name,
            source_rel,
        ) in FINAL_ARTIFACTS.items():

            source = (
                source_roots[source_name]
                / source_rel
            )

            destination = (
                out_dir
                / destination_rel
            )

            if not source.is_file():
                raise FileNotFoundError(
                    f"Expected plotting output missing: "
                    f"{source}"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                source,
                destination,
            )

    validate_outputs(out_dir)

    print()
    print(
        "PASS: canonical camera-ready visual set "
        "generated from frozen results."
    )

    for relative in FINAL_ARTIFACTS:
        print(f"  {out_dir / relative}")


if __name__ == "__main__":
    main()
