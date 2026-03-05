import pytest

def pytest_addoption(parser):
    parser.addoption("--qhttp-endpoint", action="store", default="http://localhost:8443")
    parser.addoption("--duration", action="store", default="short")
    parser.addoption("--critical-h11", action="store", default="491")

@pytest.fixture
def qhttp_endpoint(request):
    return request.config.getoption("--qhttp-endpoint")

@pytest.fixture
def duration(request):
    return request.config.getoption("--duration")

@pytest.fixture
def critical_h11(request):
    return request.config.getoption("--critical-h11")
