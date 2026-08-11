from pathlib import Path
import pandas as pd


def test_reported_delivered_water_fraction_is_physical():
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "outputs" / "hydraulic_leak_coupling" / "hydraulic_leak_timeseries.csv",
        root / "outputs" / "electrical_supply_disturbance" / "electrical_supply_disturbance_timeseries.csv",
    ]
    for path in files:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "delivered_water_fraction" not in frame.columns:
            continue
        delivered = frame["delivered_water_fraction"].dropna()
        assert (delivered >= 0.0).all()
        assert (delivered <= 1.0).all()
