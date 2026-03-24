var toggle = document.getElementById("theme-toggle");
var html = document.documentElement;

function getPreferredTheme() {
  var saved = localStorage.getItem("theme");
  if (saved) return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  html.setAttribute("data-theme", theme);
  toggle.setAttribute("aria-pressed", theme === "dark");
  toggle.title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  localStorage.setItem("theme", theme);
}

function flipTheme() {
  var current = html.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
}

toggle.addEventListener("click", flipTheme);

document.addEventListener("keydown", function (e) {
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "L") {
    e.preventDefault();
    flipTheme();
  }
});

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
  if (!localStorage.getItem("theme")) {
    applyTheme(e.matches ? "dark" : "light");
  }
});

applyTheme(getPreferredTheme());