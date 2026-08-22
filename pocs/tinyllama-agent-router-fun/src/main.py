import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from requests_data import AGENTS, REQUESTS
from router import route

MODEL = os.environ.get("MODEL", "tinyllama")
WORKERS = int(os.environ.get("WORKERS", "8"))
VERBOSE = os.environ.get("VERBOSE", "1") == "1"


def confusion(rows):
    matrix = Counter((expected, predicted) for expected, predicted, _, _ in rows)
    width = max(len(a) for a in AGENTS) + 2
    print("\nagent confusion (rows expected, columns predicted)")
    print(" " * width + "".join(f"{a[:6]:>8}" for a in AGENTS))
    for expected in AGENTS:
        cells = "".join(f"{matrix[(expected, p)]:>8}" for p in AGENTS)
        print(f"{expected:<{width}}{cells}")


def main():
    print(f"model {MODEL}, {len(REQUESTS)} labeled requests\n")
    rows = []
    start = time.perf_counter()
    for text, expected_agent, expected_sentiment in REQUESTS:
        agent, sentiment = route(text, MODEL)
        rows.append((expected_agent, agent, expected_sentiment, sentiment))
        if VERBOSE:
            mark = "ok  " if agent == expected_agent else "MISS"
            print(f"{mark} {agent:<13} {sentiment:<9} | {text[:58]}")
    serial = time.perf_counter() - start

    total = len(rows)
    agent_hits = sum(1 for e, p, _, _ in rows if e == p)
    sentiment_hits = sum(1 for _, _, e, p in rows if e == p)
    both = sum(1 for e, p, es, ps in rows if e == p and es == ps)

    print("\n" + "-" * 66)
    print(f"agent accuracy      {agent_hits / total:.2f}  ({agent_hits}/{total})")
    print(f"sentiment accuracy  {sentiment_hits / total:.2f}  ({sentiment_hits}/{total})")
    print(f"both correct        {both / total:.2f}  ({both}/{total})")
    print(f"serial latency      {serial / total * 1000:.0f} ms/request "
          f"-> {total / serial:.1f} req/sec")
    confusion(rows)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda item: route(item[0], MODEL), REQUESTS))
    parallel = time.perf_counter() - start
    print(f"\n{WORKERS} workers        {total / parallel:.1f} req/sec "
          f"({serial / parallel:.1f}x over serial)")
    print(f"extrapolated        {total / parallel * 86400 / 1_000_000:.2f}M requests/day "
          f"on this one machine, zero API cost")


if __name__ == "__main__":
    main()
