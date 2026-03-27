const pages = [
  "index.html",
  "landing-02.html",
  "landing-03.html",
  "landing-04.html",
  "landing-05.html",
  "landing-06.html",
  "landing-07.html",
  "landing-08.html",
  "landing-09.html",
  "landing-10.html"
];

const current = window.location.pathname.split("/").pop() || "index.html";

document.querySelectorAll(".nav-link").forEach((link) => {
  if (link.getAttribute("href") === current) {
    link.classList.add("active");
  }
});

document.addEventListener("keydown", (event) => {
  const index = pages.indexOf(current);
  if (index === -1) return;
  if (event.key === "ArrowRight") {
    window.location.href = pages[(index + 1) % pages.length];
  }
  if (event.key === "ArrowLeft") {
    window.location.href = pages[(index - 1 + pages.length) % pages.length];
  }
});
