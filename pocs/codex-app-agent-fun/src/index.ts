const port = Number(process.env.PORT ?? 3000)
const publicDir = new URL("../public/", import.meta.url)

const contentType = (path: string) => {
  if (path.endsWith(".html")) return "text/html"
  if (path.endsWith(".css")) return "text/css"
  if (path.endsWith(".js")) return "text/javascript"
  if (path.endsWith(".json")) return "application/json"
  return "text/plain"
}

Bun.serve({
  port,
  fetch(req) {
    const url = new URL(req.url)
    const path = url.pathname === "/" ? "/index.html" : url.pathname
    const fileUrl = new URL("." + path, publicDir)
    const file = Bun.file(fileUrl)
    return file.exists().then((exists) => {
      if (!exists) {
        return new Response("Not found", { status: 404 })
      }
      return new Response(file, { headers: { "Content-Type": contentType(path) } })
    })
  },
})

console.log(`Server running on http://localhost:${port}`)
