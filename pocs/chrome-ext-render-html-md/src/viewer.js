const supportedMarkdown = [".md", ".markdown", ".mdown", ".mkd"]
const supportedHtml = [".html", ".htm"]
const params = new URLSearchParams(location.search)
const requestedUrl = params.get("url")
const stage = document.getElementById("stage")

function validatedSourceUrl() {
  try {
    const url = new URL(requestedUrl)
    if (url.protocol !== "https:" || url.hostname !== "raw.githubusercontent.com") return null
    return url
  } catch {
    return null
  }
}

function fileType(url) {
  const path = url.pathname.toLowerCase()
  if (supportedMarkdown.some(extension => path.endsWith(extension))) return "markdown"
  if (supportedHtml.some(extension => path.endsWith(extension))) return "html"
  return ""
}

function githubUrl(url) {
  const parts = url.pathname.split("/").filter(Boolean)
  const owner = parts[0]
  const repository = parts[1]
  let reference = parts[2]
  let file = parts.slice(3)

  if (parts[2] === "refs" && ["heads", "tags"].includes(parts[3])) {
    reference = parts[4]
    file = parts.slice(5)
  }

  return `https://github.com/${owner}/${repository}/blob/${reference}/${file.join("/")}`
}

function showError(message) {
  const error = document.createElement("div")
  error.className = "github-render-error"
  error.innerHTML = `<span>Unable to render</span><strong>${DocumentRenderer.escapeHtml(message)}</strong>`
  stage.replaceChildren(error)
}

function showMarkdown(source) {
  const article = document.createElement("article")
  article.className = "github-render-paper github-render-markdown"
  article.innerHTML = DocumentRenderer.markdownToHtml(source)
  stage.replaceChildren(article)
}

function showHtml(source, url) {
  const frame = document.createElement("iframe")
  frame.className = "github-render-frame"
  frame.setAttribute("title", `Rendered ${decodeURIComponent(url.pathname.split("/").pop())}`)
  frame.src = chrome.runtime.getURL("html-viewer.html")
  frame.addEventListener("load", () => {
    frame.contentWindow.postMessage({
      type: "github-render-document",
      source,
      baseUrl: new URL(".", url).href
    }, "*")
  }, { once: true })
  stage.replaceChildren(frame)
}

function load() {
  const url = validatedSourceUrl()
  if (!url) {
    showError("The source URL is invalid.")
    return
  }

  const type = fileType(url)
  if (!type) {
    showError("This file type is not supported.")
    return
  }

  const name = decodeURIComponent(url.pathname.split("/").pop())
  document.title = `${name} · GitHub Render`
  document.getElementById("name").textContent = name
  document.getElementById("type").textContent = `Rendered ${type === "html" ? "HTML" : "Markdown"}`
  document.getElementById("github").href = githubUrl(url)
  document.getElementById("disable").addEventListener("click", () => {
    chrome.storage.sync.set({ enabled: false }, () => location.replace(url.href))
  })

  chrome.runtime.sendMessage({ type: "fetch-source", url: url.href }, response => {
    if (chrome.runtime.lastError) {
      showError(chrome.runtime.lastError.message)
      return
    }
    if (!response?.ok) {
      showError(response?.error || "GitHub did not return the file.")
      return
    }
    if (type === "markdown") showMarkdown(response.source)
    if (type === "html") showHtml(response.source, url)
  })
}

load()
