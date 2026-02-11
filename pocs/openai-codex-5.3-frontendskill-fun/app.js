const pages = [
  {
    id: 1,
    name: "Quiet Forest Intelligence",
    kicker: "calm automation",
    title: "AI that takes care of work while your team thinks big.",
    text: "A planning and execution engine tuned for focus, not noise. It reads your workflow, predicts blockers, and keeps projects moving with measured precision.",
    cta1: "Start Building",
    cta2: "Watch Flow",
    stats: ["95% faster planning", "24/7 task continuity", "SOC 2 ready"],
    cards: ["Intent Mapping", "Workflow Memory", "Auto-Resolution", "Live Risk Radar"]
  },
  {
    id: 2,
    name: "Editorial Operator",
    kicker: "premium writing ai",
    title: "Turn rough ideas into polished launches in minutes.",
    text: "Built for teams shipping weekly. Generate campaign pages, product updates, and social narratives with editorial consistency across every channel.",
    cta1: "Write Launch",
    cta2: "See Quality",
    stats: ["60 languages", "Brand tone lock", "Version history"],
    cards: ["Campaign Studio", "Voice Control", "Instant Localization", "Approval Loops"]
  },
  {
    id: 3,
    name: "Neon Command Core",
    kicker: "real-time ai ops",
    title: "Detect signal, route action, execute in real time.",
    text: "From incidents to opportunities, Command Core analyzes streams of events and orchestrates the next best move before competitors react.",
    cta1: "Open Console",
    cta2: "Read Report",
    stats: ["180ms reaction", "1.8M events/min", "Predictive routing"],
    cards: ["Live Stream Brain", "Adaptive Agents", "Escalation Rules", "Outcome Tracking"]
  },
  {
    id: 4,
    name: "Velvet Concierge",
    kicker: "luxury customer ai",
    title: "Every customer feels like your only customer.",
    text: "A conversational AI layer for high-touch brands. It adapts tone, context, and pacing to make service feel personal, elegant, and deeply human.",
    cta1: "Design Journey",
    cta2: "Hear Voices",
    stats: ["NPS +34", "Emotion-aware replies", "Cross-channel memory"],
    cards: ["Persona Craft", "Delight Triggers", "Escalation Grace", "Lifetime Insights"]
  },
  {
    id: 5,
    name: "Atlas Knowledge",
    kicker: "enterprise assistant",
    title: "Your company knowledge finally behaves like a product.",
    text: "Search, summarize, and automate actions from docs, tickets, calls, and data tables in a single secure workspace built for teams at scale.",
    cta1: "Connect Sources",
    cta2: "Tour Workspace",
    stats: ["200+ connectors", "Role-based access", "Audit trails"],
    cards: ["Unified Search", "Policy Guardrails", "Action Suggestions", "Knowledge Health"]
  },
  {
    id: 6,
    name: "Mono Precision",
    kicker: "minimal decision ai",
    title: "No clutter. Just the exact next action.",
    text: "Designed for operators who care about output. Mono Precision strips away visual noise and gives one clear recommendation tied to measurable impact.",
    cta1: "Run Analysis",
    cta2: "Open Metrics",
    stats: ["Single-step guidance", "Clean operator mode", "Decision confidence"],
    cards: ["Priority Signal", "One-Click Deploy", "Impact Forecast", "Ops Timeline"]
  },
  {
    id: 7,
    name: "Solar Growth Pilot",
    kicker: "go-to-market ai",
    title: "Launch campaigns that adapt while they run.",
    text: "Growth Pilot continuously shifts creative, audience, and spend allocation based on response quality, so your best ideas scale faster.",
    cta1: "Launch Campaign",
    cta2: "See Lift",
    stats: ["3.2x faster iteration", "Autonomous budget shift", "Live cohort insights"],
    cards: ["Creative Rotation", "Audience Discovery", "Spend Intelligence", "Retention Maps"]
  },
  {
    id: 8,
    name: "Arcade Agent Studio",
    kicker: "playful no-code ai",
    title: "Build smart assistants like snapping blocks together.",
    text: "Drag, connect, and publish AI agents for support, sales, and operations with colorful visual logic and instant channel deployment.",
    cta1: "Build Agent",
    cta2: "View Templates",
    stats: ["No-code graph builder", "One-click publish", "Multichannel output"],
    cards: ["Prompt Blocks", "Condition Paths", "Action Webhooks", "Live Agent Replay"]
  },
  {
    id: 9,
    name: "Canopy Green Compute",
    kicker: "sustainable ai cloud",
    title: "High-performance AI with a smaller carbon footprint.",
    text: "Canopy schedules training and inference on cleaner energy windows, balancing speed, cost, and sustainability with transparent reporting.",
    cta1: "Start Green Run",
    cta2: "Carbon Report",
    stats: ["42% lower emissions", "Region-aware scheduling", "Cost-energy optimizer"],
    cards: ["Eco Planner", "Carbon Ledger", "Smart Region Shift", "Efficiency Benchmarks"]
  },
  {
    id: 10,
    name: "Grand Atlas Boardroom",
    kicker: "executive ai",
    title: "Board-level insight delivered before the meeting starts.",
    text: "A strategic intelligence surface that synthesizes market movement, internal KPIs, and risk signals into clear recommendations leaders can trust.",
    cta1: "Generate Briefing",
    cta2: "Review Signals",
    stats: ["Executive digest in 5 min", "Multi-source synthesis", "Risk confidence index"],
    cards: ["Market Pulse", "KPI Narrative", "Board Summaries", "Decision Trail"]
  }
];

const app = document.getElementById("app");
const select = document.getElementById("pageSelect");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");

pages.forEach((page) => {
  const option = document.createElement("option");
  option.value = String(page.id);
  option.textContent = `${page.id}. ${page.name}`;
  select.appendChild(option);
});

function currentId() {
  const params = new URLSearchParams(window.location.search);
  const id = Number(params.get("p") || "1");
  if (!Number.isInteger(id) || id < 1 || id > 10) return 1;
  return id;
}

function updateUrl(id) {
  const params = new URLSearchParams(window.location.search);
  params.set("p", String(id));
  const url = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState({}, "", url);
}

function render(id) {
  const page = pages[id - 1];
  select.value = String(id);
  app.innerHTML = `
    <section class="site site-${id}">
      <span class="glow"></span>
      <span class="glow2"></span>
      <span class="glow3"></span>
      <article class="shell">
        <div class="hero">
          <div class="kicker">${page.kicker}</div>
          <h1>${page.title}</h1>
          <p>${page.text}</p>
          <div class="actions">
            <button class="btn">${page.cta1}</button>
            <button class="btn ghost">${page.cta2}</button>
          </div>
          <div class="stats">${page.stats.map((s) => `<span>${s}</span>`).join("")}</div>
          <div class="grid">${page.cards.map((c) => `<div class="card">${c}</div>`).join("")}</div>
        </div>
      </article>
    </section>
  `;
}

function setPage(id) {
  const bounded = ((id - 1 + pages.length) % pages.length) + 1;
  updateUrl(bounded);
  render(bounded);
}

select.addEventListener("change", () => setPage(Number(select.value)));
prevBtn.addEventListener("click", () => setPage(currentId() - 1));
nextBtn.addEventListener("click", () => setPage(currentId() + 1));
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") setPage(currentId() - 1);
  if (e.key === "ArrowRight") setPage(currentId() + 1);
});

setPage(currentId());
