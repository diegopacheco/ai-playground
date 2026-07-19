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

if (matchMedia("(hover: hover)").matches) {
    document.querySelectorAll(".card-fan, .hand-fan").forEach((group) => {
        const cards = [...group.querySelectorAll(".deck-card, .showcase-card")]
        let centers = []
        let pointer = 0
        let frame = 0
        let groupLeft = 0
        const measure = () => {
            centers = cards.map((card) => card.offsetLeft + card.offsetWidth / 2)
        }
        const render = () => {
            cards.forEach((card, index) => {
                const distance = Math.abs(pointer - centers[index])
                const influence = Math.max(0, 1 - distance / 145)
                const curve = (1 - Math.cos(influence * Math.PI)) / 2
                card.style.setProperty("--dock-scale", `${1 + curve * 0.36}`)
                card.style.setProperty("--dock-lift", `${curve * 26}px`)
                card.style.zIndex = `${Math.round(curve * 100)}`
            })
            frame = 0
        }
        group.addEventListener("pointerenter", () => {
            measure()
            groupLeft = group.getBoundingClientRect().left
            group.classList.add("dock-active")
        })
        group.addEventListener("pointermove", (event) => {
            pointer = event.clientX - groupLeft + group.scrollLeft
            if (!frame) {
                frame = requestAnimationFrame(render)
            }
        })
        group.addEventListener("pointerleave", () => {
            if (frame) {
                cancelAnimationFrame(frame)
                frame = 0
            }
            group.classList.remove("dock-active")
            requestAnimationFrame(() => {
                cards.forEach((card) => {
                    card.style.removeProperty("--dock-scale")
                    card.style.removeProperty("--dock-lift")
                    card.style.removeProperty("z-index")
                })
            })
        })
    })
}
