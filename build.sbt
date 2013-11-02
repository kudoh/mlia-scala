name := "mlia-scala"

version := "0.0.1"

scalaVersion := "2.10.2"

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies += "org.scalanlp" % "breeze_2.10" % "0.5.2"

libraryDependencies += "org.scalanlp" % "breeze-viz_2.10" % "0.5.2" 

org.scalastyle.sbt.ScalastylePlugin.Settings
