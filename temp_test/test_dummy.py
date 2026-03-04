def test_opt(pytestconfig):
    assert pytestconfig.getoption("myopt") == "custom"
