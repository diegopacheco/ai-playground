chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message.type !== "fetch-source") return false

  let url

  try {
    url = new URL(message.url)
  } catch {
    respond({ ok: false, error: "The source URL is invalid." })
    return false
  }

  if (!isAllowed(url)) {
    respond({ ok: false, error: "The source host is not allowed." })
    return false
  }

  fetch(url, { credentials: "include", redirect: "follow" })
    .then(async response => {
      if (!response.ok) throw new Error(`GitHub returned ${response.status}.`)
      return response.text()
    })
    .then(source => respond({ ok: true, source }))
    .catch(error => respond({ ok: false, error: error.message }))

  return true
})

function isAllowed(url) {
  if (url.protocol !== "https:") return false
  if (url.hostname === "raw.githubusercontent.com") return true
  return url.hostname === "github.com" && url.pathname.includes("/raw/")
}
