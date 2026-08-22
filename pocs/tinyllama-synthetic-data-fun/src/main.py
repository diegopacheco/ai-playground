import json
import os
import sys
import time

from person import PersonGenerator

OUT_FILE = os.environ.get("OUT_FILE", "people.jsonl")
COUNT = int(os.environ.get("COUNT", "200000"))
LLM_COUNT = int(os.environ.get("LLM_COUNT", "5"))
MODEL = os.environ.get("MODEL", "tinyllama")


def bench_structured():
    gen = PersonGenerator(seed=42)
    start = time.perf_counter()
    written = 0
    with open(OUT_FILE, "w") as out:
        for person in gen.many(COUNT):
            out.write(json.dumps(person))
            out.write("\n")
            written += 1
    elapsed = time.perf_counter() - start
    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print(f"structured : {written} rows in {elapsed:.2f}s -> {written / elapsed:,.0f} rows/sec")
    print(f"structured : {OUT_FILE} is {size_mb:.1f} MB")
    return elapsed


def check_determinism():
    a = list(PersonGenerator(seed=7).many(1000))
    b = list(PersonGenerator(seed=7).many(1000))
    c = list(PersonGenerator(seed=8).many(1000))
    assert a == b, "same seed must produce the same dataset"
    assert a != c, "different seeds must produce different datasets"
    emails = {p["email"] for p in a}
    assert len(emails) == len(a), "emails must be unique"
    print("checks     : deterministic by seed, emails unique")


def bench_llm():
    try:
        from ollama import chat
    except ImportError:
        print("llm        : ollama package missing, skipped")
        return
    gen = PersonGenerator(seed=99)
    people = list(gen.many(LLM_COUNT))
    start = time.perf_counter()
    bios = []
    for person in people:
        prompt = (
            f"Write one short sentence describing {person['first_name']} "
            f"{person['last_name']}, a {person['age']} year old "
            f"{person['job_title']} living in {person['city']}. "
            "Reply with the sentence only."
        )
        response = chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        bios.append((person, response.message.content.strip()))
    elapsed = time.perf_counter() - start
    print(f"llm        : {LLM_COUNT} bios in {elapsed:.2f}s -> {LLM_COUNT / elapsed:,.2f} rows/sec")
    for person, bio in bios:
        print(f"  {person['first_name']} {person['last_name']}: {bio[:160]}")
    return elapsed


def main():
    check_determinism()
    structured = bench_structured()
    llm = bench_llm()
    if llm:
        per_row_structured = structured / COUNT
        per_row_llm = llm / LLM_COUNT
        print(f"ratio      : free text costs {per_row_llm / per_row_structured:,.0f}x more time per row")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
