from lib.runners.scala3_sbt import Scala3SbtRunner


class Scala2BazelRunner(Scala3SbtRunner):
    target = "scala2-bazel"
    scala_version = "2.13"
