from lib.runners.base import BaseRunner

class KotlinRunner(BaseRunner):
    target = "kotlin"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return [{
            "name": "kotlin runner",
            "sense_check": "kotlin runner is a detection stub in this POC; no behavior probing performed.",
            "trace": "NotImplementedError: kotlin runner not implemented",
            "kind": "stub",
        }]

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return []
