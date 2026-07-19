addEventListener("message", event => {
  const message = event.data
  if (!message || message.type !== "github-render-document") return

  const parsed = new DOMParser().parseFromString(message.source, "text/html")
  parsed.querySelectorAll("base, object, embed, iframe, meta[http-equiv]").forEach(node => node.remove())
  const base = parsed.createElement("base")
  base.href = message.baseUrl
  parsed.head.prepend(base)

  document.open()
  document.write(`<!doctype html>${parsed.documentElement.outerHTML}`)
  document.close()
}, { once: true })
