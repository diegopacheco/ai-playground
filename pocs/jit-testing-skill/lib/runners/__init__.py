from lib.runners.java8 import Java8Runner
from lib.runners.java25 import Java25Runner
from lib.runners.scala3_sbt import Scala3SbtRunner
from lib.runners.scala2_bazel import Scala2BazelRunner
from lib.runners.kotlin import KotlinRunner
from lib.runners.python3 import Python3Runner
from lib.runners.python3_django import Python3DjangoRunner
from lib.runners.nodejs import NodejsRunner

REGISTRY = {
    "java8": Java8Runner,
    "java25": Java25Runner,
    "scala3-sbt": Scala3SbtRunner,
    "scala2-bazel": Scala2BazelRunner,
    "kotlin": KotlinRunner,
    "python3": Python3Runner,
    "python3-django": Python3DjangoRunner,
    "nodejs": NodejsRunner,
}

def get_runner(target):
    cls = REGISTRY.get(target)
    if cls is None:
        return None
    return cls()
