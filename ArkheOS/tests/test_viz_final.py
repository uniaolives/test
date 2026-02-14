import pytest
from arkhe.viz import VisualArchive, MultiViewTrinity

def test_visual_archive_trends():
    archive = VisualArchive()
    trends = archive.calculate_trends(duration_seconds=10.0)

    assert trends["nodes_added"] == 5
    assert trends["syzygy_growth"] == pytest.approx(0.0008, rel=1e-3)
    assert trends["projected_nodes_march_14"] > 1200000

def test_multi_view_layout():
    mvt = MultiViewTrinity()
    layout = mvt.get_layout_config()

    assert layout["primary"]["shader"] == "χ_HOLOGRAPHIC_ARK"
    assert layout["primary"]["area"] == 0.70
    assert layout["secondary_left"]["area"] == 0.15
    assert layout["secondary_right"]["area"] == 0.15

def test_visual_archive_metadata():
    archive = VisualArchive()
    meta = archive.get_video_metadata()

    assert meta["filename"] == "holographic_ark_genesis.mp4"
    assert meta["codec"] == "H.265 (HEVC)"
    assert meta["crf"] == 18
