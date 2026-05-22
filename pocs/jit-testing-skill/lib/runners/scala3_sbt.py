from lib.runners.base import BaseRunner

class Scala3SbtRunner(BaseRunner):
    target = "scala3-sbt"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return [{
            "name": "scala3-sbt runner",
            "sense_check": "scala3-sbt runner is a detection stub in this POC; no behavior probing performed.",
            "trace": "NotImplementedError: scala3-sbt runner not implemented",
            "kind": "stub",
        }]

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return []
