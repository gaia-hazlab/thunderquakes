from thunderquakes.config import REGIONS, pnwml_paths


def test_regions_have_required_keys():
    for _name, spec in REGIONS.items():
        assert "bbox" in spec and len(spec["bbox"]) == 4
        assert spec["seismic_networks"]
        assert spec["priority"] in (1, 2, 3)


def test_ok_is_priority_one():
    assert REGIONS["OK"]["priority"] == 1


def test_pnwml_paths_layout(tmp_path):
    p = pnwml_paths(tmp_path)
    assert p.comcat["metadata"].name == "metadata.csv"
    assert p.exotic["waveforms"].parent.name == "exotic"
