import pytest
from arkhe.collective_navigation import CollectiveNavigation

def test_collective_nav_init():
    nav = CollectiveNavigation()
    assert nav.initiate_nav() is True
    assert nav.participants == 24

def test_somatic_reports():
    nav = CollectiveNavigation()
    report = nav.collect_report("NODE_003", "Sinto o Toro", "Alpha sync")
    assert report["node"] == "NODE_003"
    assert len(nav.somatic_reports) == 1

def test_collective_results():
    nav = CollectiveNavigation()
    res = nav.get_results()
    assert res["Syzygy_Peak"] == 0.99
    assert res["Interface_Order"] == 0.68
    assert res["Status"] == "TELEPRESENÇA_SOMÁTICA_CONFIRMADA"
