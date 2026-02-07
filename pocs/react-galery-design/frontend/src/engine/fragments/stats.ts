export function generateStats(): string {
  const items = [
    { num: "10K+", label: "Users" },
    { num: "500+", label: "Projects" },
    { num: "99.9%", label: "Uptime" },
    { num: "24/7", label: "Support" },
  ];

  const cards = items.map((s) => `
    <div style="text-align:center;padding:32px;background:var(--bg);border-radius:var(--radius);box-shadow:var(--shadow);">
      <div style="font-size:2.5rem;font-weight:800;color:var(--primary);margin-bottom:8px;">${s.num}</div>
      <div style="font-size:1rem;color:var(--secondary);text-transform:uppercase;letter-spacing:1px;">${s.label}</div>
    </div>`).join("\n");

  return `<section style="padding:48px 32px;">
  <div style="max-width:1000px;margin:0 auto;">
    <div class="stats-row grid-4" style="display:grid;grid-template-columns:repeat(4,1fr);gap:24px;">
${cards}
    </div>
  </div>
</section>`;
}
