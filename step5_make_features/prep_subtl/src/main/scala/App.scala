import collection.JavaConverters._
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets
import io.{BufferedSource, Source}

import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.pipeline._
import play.api.libs.json.Json

object App {

  private val pipelineProps = new java.util.Properties()
  val stepsKey = "annotators"
  pipelineProps.put(stepsKey, "tokenize, ssplit, pos, lemma")

  lazy val pipeline = new StanfordCoreNLP(pipelineProps)

  def getLemmas(text: String):List[String] = getLemmas(parseText(text))

  def getLemmas(a: Annotation): List[String] =
    a.get(classOf[TokensAnnotation]).asScala.map(_.get(classOf[LemmaAnnotation])).toList

  def parseText(text: String): Annotation = {
    val document: Annotation = new Annotation(text)
    pipeline.annotate(document)
    document
  }

  val alphaNumeric = (('a' to 'z') ++ ('A' to 'Z') ++ ('0' to '9')).toSet
  def isAlphaNumeric(s: String) = s.forall(alphaNumeric.contains)

  var currentCall, previousMsg:Int = 0

  def _printStatusUpdate(map:Map[String, Int]) = {
    currentCall = currentCall + 1

    if(currentCall - previousMsg > 100000) {
      previousMsg = currentCall
      println(s"current call: $currentCall")
      println(s"map size: ${map.size}")
    }
  }

  def getLemmaCounts(lines: Iterator[String], acc:Map[String, Int] = Map.empty): Map[String, Int] = {
    _printStatusUpdate(acc)

    def update(lemmas: List[String], updated:Map[String, Int] = acc):Map[String, Int] =
      lemmas match {
        case head::tail =>
          if(isAlphaNumeric(head))
            update(tail, updated.updated(head, updated.getOrElse(head, 0) + 1))
          else
            update(tail, updated)

        case _ => updated
      }

    if(lines.hasNext)
      getLemmaCounts(lines, update(getLemmas(lines.next())))
    else acc
  }

  def main(args: Array[String]): Unit = {
    val path:String = args(0)
    val src: BufferedSource = Source.fromFile(path)(io.Codec.ISO8859)
    val counts = getLemmaCounts(src.getLines())
    println("will now convert to Json")
    val countsJs = Json.toJson(counts)
    println("write json to file")
    Files.write(Paths.get("counts.json"),
      Json.stringify(countsJs).getBytes(StandardCharsets.ISO_8859_1))
  }
  
}
