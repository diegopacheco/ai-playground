ThisBuild / scalaVersion := "3.7.3"
ThisBuild / organization := "bp"
ThisBuild / version := "1.0.0"

lazy val springBoot = "4.0.6"

lazy val root = (project in file("."))
  .settings(
    name := "java25-scala3-sbt-sb4",
    libraryDependencies ++= Seq(
      "org.springframework.boot" % "spring-boot-starter-web" % springBoot,
      "org.springframework.boot" % "spring-boot-starter-actuator" % springBoot,
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),
    javacOptions ++= Seq("-source", "25", "-target", "25"),
    Compile / mainClass := Some("bp.App"),
    Test / fork := true,
    Test / parallelExecution := false
  )
