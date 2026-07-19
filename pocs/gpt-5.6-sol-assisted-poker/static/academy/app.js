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
        group.addEventListener("pointermove", (event) => {
            const pointer = event.clientX - group.getBoundingClientRect().left + group.scrollLeft
            cards.forEach((card) => {
                const center = card.offsetLeft + card.offsetWidth / 2
                const distance = pointer - center
                const influence = Math.max(0, 1 - Math.abs(distance) / 175)
                const curve = influence * influence
                card.style.setProperty("--dock-scale", `${1 + curve * 0.56}`)
                card.style.setProperty("--dock-lift", `${curve * 38}px`)
                card.style.setProperty("--dock-shift", `${-Math.sign(distance) * curve * 12}px`)
                card.style.setProperty("--dock-turn", `${distance / 175 * curve * 5}deg`)
                card.style.setProperty("--dock-shadow-y", `${13 + curve * 22}px`)
                card.style.setProperty("--dock-shadow-blur", `${18 + curve * 30}px`)
                card.style.zIndex = `${Math.round(curve * 100)}`
            })
        })
        group.addEventListener("pointerleave", () => {
            cards.forEach((card) => {
                card.style.removeProperty("--dock-scale")
                card.style.removeProperty("--dock-lift")
                card.style.removeProperty("--dock-shift")
                card.style.removeProperty("--dock-turn")
                card.style.removeProperty("--dock-shadow-y")
                card.style.removeProperty("--dock-shadow-blur")
                card.style.removeProperty("z-index")
            })
        })
    })
}
