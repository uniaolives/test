def pytest_addoption(parser):
    parser.addoption("--myopt", action="store", default="val")
