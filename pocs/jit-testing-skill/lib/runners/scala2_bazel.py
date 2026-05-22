from lib.runners.base import BaseRunner

class Scala2BazelRunner(BaseRunner):
    target = "scala2-bazel"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return [{
            "name": "scala2-bazel runner",
            "sense_check": "scala2-bazel runner is a detection stub in this POC; no behavior probing performed.",
            "trace": "NotImplementedError: scala2-bazel runner not implemented",
            "kind": "stub",
        }]

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return []
