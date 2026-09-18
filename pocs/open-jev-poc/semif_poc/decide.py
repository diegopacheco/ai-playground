from semif_phase1.core import validate_row
from semif_phase1.direct import score


def summarize(result: dict) -> dict:
    ranked = sorted(
        ({"id": option, "probability": probability} for option, probability in zip(result["option_ids"], result["probabilities"])),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "id": result["id"],
        "decision": ranked[0]["id"],
        "ranked": ranked,
        "input_tokens": result["input_tokens"],
        "forward_ms": round(result["forward_seconds"] * 1000, 1),
    }


def decide(model, tokenizer, metadata: dict, row: dict, scorer=score) -> dict:
    validate_row(row)
    return summarize(scorer(model, tokenizer, row, metadata))
