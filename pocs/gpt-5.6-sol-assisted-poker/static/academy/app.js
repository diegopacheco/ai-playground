document.addEventListener("submit", async (event) => {
    const form = event.target
    const button = form.querySelector("button[type='submit']")
    if (button) {
        button.classList.add("working")
        button.setAttribute("aria-busy", "true")
    }
    const playArea = form.closest("[data-live-area]")
    if (!playArea) {
        return
    }
    event.preventDefault()
    playArea.querySelectorAll(".table-error").forEach((message) => message.remove())
    const scrollPosition = window.scrollY
    try {
        const target = new URL(form.getAttribute("action"), window.location.href)
        const response = await fetch(target, {
            method: "POST",
            body: new FormData(form),
            headers: {"X-Requested-With": "XMLHttpRequest"}
        })
        if (!response.ok) {
            throw new Error("The table could not complete that move.")
        }
        const page = new DOMParser().parseFromString(await response.text(), "text/html")
        const nextArea = page.querySelector(`#${playArea.id}`)
        if (!nextArea) {
            throw new Error("The table returned an invalid response.")
        }
        playArea.replaceWith(nextArea)
        history.replaceState({}, "", response.url)
        window.scrollTo(0, scrollPosition)
    } catch (error) {
        button?.classList.remove("working")
        button?.removeAttribute("aria-busy")
        const message = document.createElement("p")
        message.className = "table-error"
        message.textContent = error.message
        playArea.prepend(message)
    }
})
