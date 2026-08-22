import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors import detect_hybrid, detect_llm, detect_regex
from samples import SAMPLES

MODEL = os.environ.get("MODEL", "tinyllama")
VERBOSE = os.environ.get("VERBOSE", "1") == "1"


class Score:
    def __init__(self, name):
        self.name = name
        self.tp = self.fp = self.fn = self.exact = 0
        self.seconds = 0.0

    def add(self, expected, predicted, seconds):
        self.tp += len(expected & predicted)
        self.fp += len(predicted - expected)
        self.fn += len(expected - predicted)
        self.exact += expected == predicted
        self.seconds += seconds

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def line(self, total):
        return (f"{self.name:<18} precision {self.precision:.2f}  recall {self.recall:.2f}  "
                f"f1 {self.f1:.2f}  exact {self.exact:>2}/{total}  "
                f"{self.seconds / total * 1000:>6.1f} ms/text")


DETECTORS = [
    ("regex", lambda text: detect_regex(text)),
    ("llm all labels", lambda text: detect_llm(text, MODEL)),
    ("regex + llm name", lambda text: detect_hybrid(text, MODEL)),
]


def main():
    print(f"model {MODEL}, {len(SAMPLES)} labeled texts\n")
    scores = [Score(name) for name, _ in DETECTORS]

    for text, expected in SAMPLES:
        predictions = []
        for score, (_, detect) in zip(scores, DETECTORS):
            start = time.perf_counter()
            predicted = detect(text)
            score.add(expected, predicted, time.perf_counter() - start)
            predictions.append(predicted)
        if VERBOSE:
            print(text)
            print(f"  expected           {sorted(expected) or ['-']}")
            for (name, _), predicted in zip(DETECTORS, predictions):
                mark = "ok  " if predicted == expected else "MISS"
                print(f"  {name:<18} {mark} {sorted(predicted) or ['-']}")
            print()

    total = len(SAMPLES)
    print("-" * 78)
    for score in scores:
        print(score.line(total))
    best = max(scores, key=lambda s: s.f1)
    print(f"\nbest f1: {best.name}")
    print(f"throughput at {best.name}: {total / best.seconds:.1f} texts/sec, single process")


if __name__ == "__main__":
    main()
