const toggle = document.getElementById("enabled")
const status = document.getElementById("status")
const note = document.getElementById("note")

function update(enabled) {
  toggle.checked = enabled
  status.textContent = enabled ? "Enabled" : "Disabled"
  note.textContent = enabled
    ? "Supported GitHub files open in a clean reading view."
    : "GitHub files use their standard source view."
}

chrome.storage.sync.get({ enabled: true }, settings => update(settings.enabled))

toggle.addEventListener("change", () => {
  chrome.storage.sync.set({ enabled: toggle.checked })
  update(toggle.checked)
})
