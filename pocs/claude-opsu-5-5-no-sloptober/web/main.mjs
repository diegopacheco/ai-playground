import { startScene } from "./scene.mjs";
import { score } from "./slop.mjs";

const SKILL = ".claude/skills/no-slop-pr/";
const state = { scroll: 0, shockAt: -1e9, pledge: 0 };
const $ = (sel) => document.querySelector(sel);

function trackScroll() {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  state.scroll = max > 0 ? Math.min(Math.max(window.scrollY / max, 0), 1) : 0;
}

function countdown() {
  const now = new Date();
  const start = new Date(now.getFullYear(), 9, 1);
  const end = new Date(now.getFullYear(), 10, 1);
  const out = $("#countdown");
  if (now >= end) {
    out.textContent = "October is over. You did it by yourself.";
    return;
  }
  if (now >= start) {
    out.textContent = `Day ${now.getDate()} of 31. Keep your hands on the keyboard.`;
    return;
  }
  const s = Math.floor((start - now) / 1000);
  const pad = (n) => String(n).padStart(2, "0");
  out.textContent = `Starts in ${Math.floor(s / 86400)}d ${pad(Math.floor(s / 3600) % 24)}h ${pad(Math.floor(s / 60) % 60)}m ${pad(s % 60)}s`;
}

function reveal() {
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add("in")),
    { threshold: 0.15 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

function renderMeter(result) {
  $("#meter-rest").style.width = `${100 - result.score}%`;
  $("#meter-score").textContent = result.score;
  const verdict = $("#meter-verdict");
  verdict.textContent = result.verdict;
  verdict.dataset.verdict = result.verdict;
  $("#meter-hits").replaceChildren(
    ...(result.hits.length ? result.hits : [{ label: "No slop signals found", points: 0 }]).map((hit) => {
      const li = document.createElement("li");
      li.innerHTML = `<span></span><b></b>`;
      li.firstChild.textContent = hit.label;
      li.lastChild.textContent = hit.points ? `+${hit.points}` : "";
      return li;
    })
  );
}

async function meter() {
  const rules = await (await fetch(SKILL + "rules.json")).json();
  const input = $("#pr-text");
  const run = () => renderMeter(score(rules, input.value));
  input.addEventListener("input", run);
  document.querySelectorAll("[data-fixture]").forEach((btn) =>
    btn.addEventListener("click", async () => {
      input.value = await (await fetch(SKILL + "tests/fixtures/" + btn.dataset.fixture)).text();
      run();
    })
  );
  run();
}

function animatePledge(target) {
  const from = state.pledge;
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min((now - t0) / 2200, 1);
    state.pledge = from + (target - from) * (1 - Math.pow(1 - k, 3));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  state.shockAt = performance.now();
}

function remember(value) {
  try {
    if (value) localStorage.setItem("no-sloptober-pledge", "1");
    else localStorage.removeItem("no-sloptober-pledge");
  } catch {}
}

function recalled() {
  try {
    return localStorage.getItem("no-sloptober-pledge") === "1";
  } catch {
    return false;
  }
}

function pledge() {
  const btn = $("#pledge");
  const set = (on) => {
    document.body.classList.toggle("pledged", on);
    btn.textContent = on ? "Pledged. I do by MYSELF." : "I do by MYSELF";
  };
  if (recalled()) {
    set(true);
    state.pledge = 1;
  }
  btn.addEventListener("click", () => {
    const on = !document.body.classList.contains("pledged");
    set(on);
    remember(on);
    animatePledge(on ? 1 : 0);
  });
}

function shockwave() {
  window.addEventListener("pointerdown", (e) => {
    if (e.target.closest("a, button, textarea, input")) return;
    state.shockAt = performance.now();
  });
}

try {
  startScene($("#scene"), state);
} catch (err) {
  document.body.classList.add("no-webgl");
  console.error(err);
}
window.addEventListener("scroll", trackScroll, { passive: true });
trackScroll();
countdown();
setInterval(countdown, 1000);
reveal();
pledge();
shockwave();
meter();
