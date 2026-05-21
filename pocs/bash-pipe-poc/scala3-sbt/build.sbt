ThisBuild / scalaVersion := "3.7.3"
ThisBuild / organization := "bp"
ThisBuild / version := "1.0.0"

lazy val root = (project in file("."))
  .settings(
    name := "scala3-sbt",
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "cask" % "0.10.2",
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),
    Compile / mainClass := Some("bp.App"),
    Test / fork := true,
    Test / parallelExecution := false
  )
