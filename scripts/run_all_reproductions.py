import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(f"\n=== Running: {' '.join(command)} ===")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    commands = [
        [sys.executable, "scripts/finite_string_modes.py"],
        [sys.executable, "scripts/compact_support_qnm_scan.py", "--fast"],
        [sys.executable, "scripts/time_domain_wave_evolution.py"],
        [sys.executable, "scripts/bromwich_poles_diagram.py"],
        [sys.executable, "scripts/schwarzschild_potentials.py"],
        [sys.executable, "scripts/wkb_schwarzschild_qnm.py"],
        [sys.executable, "scripts/tov_polytrope.py"],
        [sys.executable, "scripts/ringdown_fit_demo.py"],
    ]
    for command in commands:
        run(command)
    print("\nAll reproduction scripts completed.")


if __name__ == "__main__":
    main()

