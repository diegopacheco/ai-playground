package bp

object App extends cask.MainRoutes {

  override def host: String = "127.0.0.1"
  override def port: Int =
    sys.props.get("app.port").map(_.toInt)
      .orElse(sys.env.get("PORT").map(_.toInt))
      .getOrElse(8085)

  @cask.get("/health")
  def health(): cask.Response[String] =
    cask.Response("""{"status":"ok"}""", 200, Seq("Content-Type" -> "application/json"))

  @cask.get("/")
  def root(): String = "scala3-sbt"

  initialize()
}
