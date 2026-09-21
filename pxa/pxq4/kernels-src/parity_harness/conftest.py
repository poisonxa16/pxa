"""pytest glue.

The gates are plain functions so they run with `python -m parity_harness.run_gates` on a
box with nothing but numpy.  When pytest IS available these hooks make it collect them
too: Skip becomes pytest.skip, and the optional `real`/`report` parameters get fixtures.
"""

import pytest

from .test_a_dequant import Skip


@pytest.fixture
def real(pytestconfig):
    """--real-dir=DIR on the pytest command line loads fixtures from the artifact."""
    path = pytestconfig.getoption("--real-dir", default=None)
    if not path:
        return None
    from . import fixtures
    return fixtures.load_raw_dir(path)


@pytest.fixture
def report():
    return []


def pytest_addoption(parser):
    parser.addoption("--real-dir", action="store", default=None,
                     help="fixture directory written by parity_harness.extract_raw")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    try:
        outcome.get_result()
    except Skip as e:
        pytest.skip(str(e))
