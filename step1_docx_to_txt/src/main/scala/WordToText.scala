import java.io._

import org.apache.tika.detect.DefaultDetector
import org.apache.tika.io.TikaInputStream
import org.apache.tika.metadata.Metadata
import org.apache.tika.parser.AutoDetectParser
import org.apache.tika.parser.ParseContext
import org.apache.tika.parser.Parser
import org.apache.tika.sax.BodyContentHandler

object WordToText {

  val docxRegex = """(.*)(\.docx)""".r

  def _docxContent(file:File): String = {
    val url = file.toURI.toURL

    val metadata = new Metadata()
    val input = TikaInputStream.get(url, metadata)

    val outputStream = new ByteArrayOutputStream()

    val outputWriter = new OutputStreamWriter(outputStream, "UTF-8")

    val handler = new BodyContentHandler(outputWriter)

    val p = new AutoDetectParser(new DefaultDetector())
    val context = new ParseContext()
    context.set(classOf[Parser], p)
    p.parse(input, handler, metadata, context)

    input.close()
    outputStream.flush()

    outputStream.toString("UTF-8")

  }


  def docxToTxt(docxFile: File, dir:Option[String] = None)= {

    val content = _docxContent(docxFile)

    val txtFilePath = dir match {
      case Some(d) => d
      case _ => docxFile.getParent
    }

    val txtFileName = docxFile.getName match {
      case docxRegex(nameNoExt, _) => nameNoExt + ".txt"
    }

    val txtFile = new File(txtFilePath, txtFileName)
    val writer = new FileWriter(txtFile)

    writer.write(content)
    writer.close()
  }

  def convertDocxFiles(docxPath: String, txtPath:Option[String] = None) = {
    val docxDir = new File(docxPath)
    for {
      f <- docxDir.listFiles()
      if f.getName.matches(".*docx$")
    } {
      print(s"about to convert ${f.getName}")
      docxToTxt(f, txtPath)
    }
  }

}
