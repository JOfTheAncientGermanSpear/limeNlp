name := """prep-subtl"""

version := "1.0-SNAPSHOT"

scalaVersion := "2.11.1"

resolvers += "Typesafe Repo" at "http://repo.typesafe.com/typesafe/releases/"

libraryDependencies ++= Seq(
  "org.apache.tika" % "tika-parsers" % "1.7",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.5.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.5.0" classifier "models",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.5.0" classifier "sources",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.5.0" classifier "javadoc",
  "com.typesafe.play" %% "play-json" % "2.3.4"
)
