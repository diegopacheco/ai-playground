let loaded = false

addEventListener("message", event => {
  const message = event.data
  if (loaded || event.source !== parent || !message || message.type !== "github-render-document") return

  let baseUrl
  try {
    const url = new URL(message.baseUrl)
    if (url.protocol !== "https:" || !["github.com", "raw.githubusercontent.com"].includes(url.hostname)) return
    baseUrl = url.href
  } catch {
    return
  }

  loaded = true

  const parsed = new DOMParser().parseFromString(message.source, "text/html")
  parsed.querySelectorAll("base, object, embed, iframe, meta[http-equiv]").forEach(node => node.remove())
  const base = parsed.createElement("base")
  base.href = baseUrl
  parsed.head.prepend(base)

  document.open()
  document.write(`<!doctype html>${parsed.documentElement.outerHTML}`)
  document.close()
})
