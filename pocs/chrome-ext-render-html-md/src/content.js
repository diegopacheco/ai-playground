const ROOT_ID = "github-render-root"
const SUPPORTED_MARKDOWN = [".md", ".markdown", ".mdown", ".mkd"]
const SUPPORTED_HTML = [".html", ".htm"]
let currentUrl = ""
let dismissedUrl = ""
let requestNumber = 0
let scheduled = false

function fileType(url = location.href) {
  const path = new URL(url).pathname.toLowerCase()
  if (SUPPORTED_MARKDOWN.some(extension => path.endsWith(extension))) return "markdown"
  if (SUPPORTED_HTML.some(extension => path.endsWith(extension))) return "html"
  return ""
}

function sourceUrl() {
  const url = new URL(location.href)
  if (url.hostname === "github.com") url.pathname = url.pathname.replace("/blob/", "/raw/")
  return url.href
}

function fileName() {
  return decodeURIComponent(new URL(location.href).pathname.split("/").pop() || "Document")
}

function removeViewer() {
  document.getElementById(ROOT_ID)?.remove()
  document.documentElement.classList.remove("github-render-open")
}

function createViewer(type) {
  removeViewer()
  const root = document.createElement("section")
  root.id = ROOT_ID
  root.innerHTML = `
    <header class="github-render-bar">
      <div class="github-render-identity">
        <span class="github-render-mark">GR</span>
        <div>
          <span class="github-render-label">Rendered ${type === "markdown" ? "Markdown" : "HTML"}</span>
          <strong>${DocumentRenderer.escapeHtml(fileName())}</strong>
        </div>
      </div>
      <div class="github-render-actions">
        <button class="github-render-source" type="button">View source</button>
        <a class="github-render-raw" href="${DocumentRenderer.escapeHtml(sourceUrl())}" target="_blank" rel="noreferrer">Open raw</a>
      </div>
    </header>
    <main class="github-render-stage">
      <div class="github-render-loading"><span></span><p>Typesetting document</p></div>
    </main>`
  root.querySelector(".github-render-source").addEventListener("click", () => {
    dismissedUrl = location.href
    removeViewer()
  })
  document.body.append(root)
  document.documentElement.classList.add("github-render-open")
  return root
}

function showMarkdown(stage, source) {
  const article = document.createElement("article")
  article.className = "github-render-paper github-render-markdown"
  article.innerHTML = DocumentRenderer.markdownToHtml(source)
  stage.replaceChildren(article)
}

function showHtml(stage, source) {
  const frame = document.createElement("iframe")
  frame.className = "github-render-frame"
  frame.setAttribute("sandbox", "allow-scripts")
  frame.setAttribute("title", `Rendered ${fileName()}`)
  frame.src = chrome.runtime.getURL("html-viewer.html")
  frame.addEventListener("load", () => {
    frame.contentWindow.postMessage({
      type: "github-render-document",
      source,
      baseUrl: new URL(".", sourceUrl()).href
    }, "*")
  }, { once: true })
  stage.replaceChildren(frame)
}

function showError(stage, message) {
  const error = document.createElement("div")
  error.className = "github-render-error"
  error.innerHTML = `<span>Unable to render</span><strong>${DocumentRenderer.escapeHtml(message)}</strong><button type="button">Return to source</button>`
  error.querySelector("button").addEventListener("click", () => {
    dismissedUrl = location.href
    removeViewer()
  })
  stage.replaceChildren(error)
}

async function renderPage() {
  scheduled = false
  const url = location.href
  const type = fileType(url)

  if (!type || dismissedUrl === url) {
    if (currentUrl !== url) removeViewer()
    currentUrl = url
    return
  }

  const settings = await chrome.storage.sync.get({ enabled: true })
  if (!settings.enabled) {
    currentUrl = url
    removeViewer()
    return
  }
  if (url === currentUrl && document.getElementById(ROOT_ID)) return

  currentUrl = url
  const activeRequest = ++requestNumber
  const viewer = createViewer(type)
  const stage = viewer.querySelector(".github-render-stage")

  chrome.runtime.sendMessage({ type: "fetch-source", url: sourceUrl() }, response => {
    if (activeRequest !== requestNumber || !viewer.isConnected) return
    if (chrome.runtime.lastError) {
      showError(stage, chrome.runtime.lastError.message)
      return
    }
    if (!response?.ok) {
      showError(stage, response?.error || "GitHub did not return the file.")
      return
    }
    if (type === "markdown") showMarkdown(stage, response.source)
    if (type === "html") showHtml(stage, response.source)
  })
}

function scheduleRender() {
  if (scheduled) return
  scheduled = true
  setTimeout(renderPage, 80)
}

chrome.storage.onChanged.addListener(changes => {
  if (!changes.enabled) return
  if (changes.enabled.newValue) {
    currentUrl = ""
    dismissedUrl = ""
    scheduleRender()
  } else {
    removeViewer()
  }
})

document.addEventListener("turbo:load", scheduleRender)
document.addEventListener("pjax:end", scheduleRender)
window.addEventListener("popstate", scheduleRender)
new MutationObserver(() => {
  if (location.href !== currentUrl) scheduleRender()
}).observe(document.documentElement, { childList: true, subtree: true })
scheduleRender()
