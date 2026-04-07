# QNM Review Study and Reproduction Pack

## Overview

This workspace turns the review paper

- `9909058v1 QNM圣经.pdf`

into a combined study package:

- structured lecture notes aligned to the original paper
- runnable numerical scripts for the key toy problems and core physics ideas
- output figures that make the abstract mathematics concrete

The goal is not only to "reproduce numbers", but to build a full understanding of:

- what QNMs are
- why they appear in open wave systems
- how black-hole and stellar QNMs differ
- how numerical methods recover QNM spectra
- how ringdown data connect to observable source parameters

## Main Documents

- [QNM论文多轮交互讲解与能力训练方案.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/QNM论文多轮交互讲解与能力训练方案.md)
- [第00到01轮_概念基础讲义.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/docs/第00到01轮_概念基础讲义.md)
- [第2轮_Laplace变换_Green函数与QNM极点_原论文讲解.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/第2轮_Laplace变换_Green函数与QNM极点_原论文讲解.md)
- [第03轮_晚时间行为_tail与QNM适用边界.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/docs/第03轮_晚时间行为_tail与QNM适用边界.md)
- [第04到06轮_黑洞QNM讲义.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/docs/第04到06轮_黑洞QNM讲义.md)
- [第07到11轮_中子星QNM讲义.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/docs/第07到11轮_中子星QNM讲义.md)
- [第12到14轮_激发探测数值方法与统一视角.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/docs/第12到14轮_激发探测数值方法与统一视角.md)
- [数值复现总指南.md](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/docs/数值复现总指南.md)

## Scripts

- [finite_string_modes.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/finite_string_modes.py)
  Finite-interval normal modes.
- [compact_support_qnm_scan.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/compact_support_qnm_scan.py)
  Laplace-domain QNM scan for a compact-support toy potential.
- [time_domain_wave_evolution.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/time_domain_wave_evolution.py)
  Time-domain wave evolution showing burst and ringdown.
- [schwarzschild_potentials.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/schwarzschild_potentials.py)
  Regge-Wheeler and Zerilli potentials.
- [wkb_schwarzschild_qnm.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/wkb_schwarzschild_qnm.py)
  First-order WKB estimate of Schwarzschild QNMs.
- [tov_polytrope.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/tov_polytrope.py)
  Simple TOV integration for a relativistic polytrope.
- [ringdown_fit_demo.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/ringdown_fit_demo.py)
  Synthetic noisy ringdown fitting.
- [bromwich_poles_diagram.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/bromwich_poles_diagram.py)
  Conceptual Bromwich contour and pole sketch.
- [run_all_reproductions.py](C:/Users/97747/GW%20QNM%20EMRI%20PROJECT/9909/scripts/run_all_reproductions.py)
  Convenience runner for the full educational script set.

## Quick Start

Run these from the workspace root:

```powershell
python scripts/finite_string_modes.py
python scripts/compact_support_qnm_scan.py --fast
python scripts/time_domain_wave_evolution.py
python scripts/schwarzschild_potentials.py
python scripts/wkb_schwarzschild_qnm.py
python scripts/tov_polytrope.py
python scripts/ringdown_fit_demo.py
```

The scripts write figures and summary text into:

- `outputs/`

## Suggested Learning Order

1. Read the master roadmap.
2. Work through the round-2 note on Laplace transforms and QNM poles.
3. Run the compact-support scan and time-domain evolution scripts.
4. Read the black-hole lecture note and run the Schwarzschild scripts.
5. Read the neutron-star lecture note and run the TOV script.
6. Finish with the detection, fitting, and numerical-method note.

## Verification Status

Representative scripts are designed to run with:

- Python 3.11
- `numpy`
- `scipy`
- `matplotlib`

Some scripts are educational toy models rather than high-precision production solvers. The point is to expose the mathematics and physics transparently.
