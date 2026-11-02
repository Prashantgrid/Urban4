"""Regression checks against the final manuscript's retained numerical evidence."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_retained_study_reproduces_the_published_sensitivities(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, str(ROOT / "code/reproduce_study.py"),
                    "--output", str(tmp_path)], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True, timeout=120)
    for name in ("base_load_sensitivity.csv", "transformer_capacity_sensitivity.csv"):
        reference = pd.read_csv(ROOT / "results/heat_power_case" / name)
        actual = pd.read_csv(tmp_path / "results/heat_power_case" / name)
        pd.testing.assert_frame_equal(actual, reference, check_exact=False, rtol=1e-10, atol=1e-10)
    for name in ("Fig03_Design_Relations.pdf", "Fig06_Seattle_Routes.pdf", "Fig07_Heat_Power_Case.pdf"):
        assert (tmp_path / "figures" / name).stat().st_size > 1_000


def test_routine_fault_claim_holds_for_both_repair_time_assumptions() -> None:
    risk = pd.read_csv(ROOT / "results/water_power_case/water_risk_by_storage.csv")
    selected = risk.loc[(risk.grid == "R1") & (risk.backup == "M0")
                        & np.isclose(risk.storage_h, 24.0)]
    assert set(selected.t_rep_h) == {5.0, 24.0}
    assert selected.E_pe_hours_per_a.between(0.0, 50.0, inclusive="left").all()


def test_single_line_outage_results_retain_the_reported_cross_sector_impact() -> None:
    events = pd.read_csv(ROOT / "results/water_power_case/mv_line_contingencies.csv")
    assert events.line.nunique() == len(events) == 407
    branch = events.loc[events.e_loads_lost == 33]
    critical = branch.loc[branch.M0_w_pop_equiv > 68_000]
    assert len(critical) == 1
    assert int(critical.iloc[0].e_rank) == 347
    assert events.M2_w_pop_equiv.abs().max() < 1e-9
