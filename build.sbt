name := "mlia-scala"

version := "0.0.1"

scalaVersion := "2.10.2"

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies += "org.scalanlp" % "breeze-math_2.10" % "0.5-SNAPSHOT"

libraryDependencies += "org.scalanlp" % "breeze-viz_2.10" % "0.5" exclude("com.github.fommil.netlib", "all")

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1"

org.scalastyle.sbt.ScalastylePlugin.Settings
