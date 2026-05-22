from lib.runners.java8 import Java8Runner

class Java25Runner(Java8Runner):
    target = "java25"

    def _javac_args(self):
        return ["-source", "25", "-target", "25"]
