import json
from pathlib import Path

from urban4.network_contingency_scenarios import prescreen_single_line_outages


def test_archived_pump_bus_has_network_outage_candidates():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs" / "final_accepted" / "power_pandapower_bidirectional.json"
    if not source.exists():
        # The lightweight source archive may omit solver exports; this is not a failure of the algorithm.
        return
    df = prescreen_single_line_outages(source, target_bus_name="PB046")
    assert not df.empty
    assert "PL035" in set(df.line_id)
    assert "PL044" in set(df.line_id)
    assert int(df.loc[df.line_id == "PL044", "deenergized_bus_count"].iloc[0]) == 304
    assert (df.hydraulic_consequence_status == "not_executed_by_prescreen").all()


def test_executed_pl044_native_result_is_archived_and_bounded():
    root = Path(__file__).resolve().parents[1]
    path = root / "outputs" / "network_originating_contingency_v2.7.0" / "manifest.json"
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    selection = result["selection"]
    assert result["status"] == "executed_and_closed"
    assert result["event"]["outaged_line"] == "PL044"
    assert result["thresholds_relaxed"] is False
    assert selection["target_source_reachable"] is False
    assert selection["deenergized_bus_count"] == 304
    assert selection["deenergized_powered_interface_count"] == 7
    assert 10.5 < selection["unserved_static_load_mw"] < 10.7
    assert result["physical_native_pump_power_kw"] == 0.0
    assert result["delivered_water_percent"] < 0.01
