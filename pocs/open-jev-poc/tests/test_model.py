import json

import pytest

from semif_poc.decide import decide
from semif_poc.model import load
from semif_poc.server import SAMPLES

EXPECTED = {
    "route-ticket": "account_access",
    "deploy-ok": "yes",
    "retry-call": "retry",
    "action-firewall": "block",
}


@pytest.fixture(scope="module")
def loaded():
    return load()


@pytest.mark.parametrize("row", json.loads(SAMPLES), ids=lambda row: row["id"])
def test_real_4b_model_makes_the_obvious_decision(loaded, row):
    result = decide(*loaded, row)
    assert result["decision"] == EXPECTED[row["id"]]
    assert abs(sum(r["probability"] for r in result["ranked"]) - 1) < 1e-6


def test_flipping_the_evidence_flips_the_decision(loaded):
    row = next(r for r in json.loads(SAMPLES) if r["id"] == "action-firewall")
    safe = {**row, "state": row["state"].replace("rm -rf /var/lib/postgres", "ls /var/log")}
    assert decide(*loaded, safe)["decision"] == "allow"
