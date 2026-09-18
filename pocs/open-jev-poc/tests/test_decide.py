import pytest

from semif_poc.decide import decide

ROW = {
    "id": "r1",
    "state": "Card was charged twice.",
    "question": "Which queue?",
    "options": [{"id": "access", "description": "Access."}, {"id": "billing", "description": "Billing."}],
}


def fake_scorer(probabilities):
    def scorer(model, tokenizer, row, metadata):
        return {
            "id": row["id"],
            "option_ids": [o["id"] for o in row["options"]],
            "probabilities": probabilities,
            "input_tokens": 42,
            "forward_seconds": 0.25,
        }

    return scorer


def test_decision_is_most_probable_option_not_first_listed():
    result = decide(None, None, {}, ROW, scorer=fake_scorer([0.2, 0.8]))
    assert result["decision"] == "billing"
    assert [r["id"] for r in result["ranked"]] == ["billing", "access"]


def test_probabilities_are_kept_so_callers_can_threshold():
    result = decide(None, None, {}, ROW, scorer=fake_scorer([0.55, 0.45]))
    assert result["ranked"][0]["probability"] == 0.55
    assert result["forward_ms"] == 250.0


def never_called(*_):
    raise AssertionError("model must not run on an invalid row")


@pytest.mark.parametrize(
    "row",
    [
        {**ROW, "options": [{"id": "only", "description": "One."}]},
        {**ROW, "options": [{"id": "a", "description": "A."}, {"id": "a", "description": "B."}]},
        {**ROW, "state": ""},
        {k: v for k, v in ROW.items() if k != "question"},
    ],
)
def test_invalid_rows_are_rejected_before_spending_a_forward_pass(row):
    with pytest.raises(ValueError):
        decide(None, None, {}, row, scorer=never_called)
