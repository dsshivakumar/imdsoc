import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from imdsoc.synthetic import build_network  # noqa: E402
from imdsoc.training import fit_predictors, log_history  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    net = build_network(seed=7)
    suite, diag = fit_predictors(log_history(net, days=6, plans_per_day=500), net)
    net.today = 0
    return net, suite, diag


@pytest.fixture()
def net():
    n = build_network(seed=7)
    n.today = 0
    return n
