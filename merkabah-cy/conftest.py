import pytest

def pytest_addoption(parser):
    parser.addoption("--qhttp-endpoint", action="store", default="http://localhost:8443", help="QHTTP server endpoint")
    parser.addoption("--duration", action="store", default="short", help="Duration of the test")
    parser.addoption("--critical-h11", action="store", default="491", help="Critical H11 value for safety checks")

@pytest.fixture
def qhttp_endpoint(request):
    return request.config.getoption("--qhttp-endpoint")

@pytest.fixture
def duration(request):
    return request.config.getoption("--duration")

@pytest.fixture
def critical_h11(request):
    return request.config.getoption("--critical-h11")
