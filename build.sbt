name := "mlia-scala"

version := "0.0.1"

scalaVersion := "2.10.2"

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies += "org.scalanlp" % "breeze-math_2.10" % "0.5-SNAPSHOT"

libraryDependencies += "org.scalanlp" % "breeze-viz_2.10" % "0.5" exclude("com.github.fommil.netlib", "all")

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1"

libraryDependencies += "org.specs2" % "specs2_2.10" % "2.1" % "test"

libraryDependencies += "org.scalacheck" % "scalacheck_2.10.0" % "1.10.0" % "test"

org.scalastyle.sbt.ScalastylePlugin.Settings
