export function generateCardGrid(): string {
  const cards = Array.from({ length: 6 }, (_, i) => `
    <div class="card" style="background:var(--bg);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden;">
      <div style="height:8px;background:var(--primary);"></div>
      <div style="padding:24px;">
        <h3 style="margin-bottom:8px;color:var(--text);">Card ${i + 1}</h3>
        <p style="color:var(--secondary);font-size:0.9rem;margin-bottom:16px;">This is a brief description of what this card represents and its purpose.</p>
        <a href="#" style="color:var(--primary);font-weight:600;">Learn More</a>
      </div>
    </div>`).join("\n");

  return `<section style="padding:48px 32px;">
  <div class="container" style="max-width:1200px;margin:0 auto;">
    <h2 style="text-align:center;margin-bottom:32px;color:var(--text);">Our Offerings</h2>
    <div class="grid-3" style="display:grid;grid-template-columns:repeat(3,1fr);gap:24px;">
${cards}
    </div>
  </div>
</section>`;
}
