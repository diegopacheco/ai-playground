export function generateGallery(): string {
  const colors = [210, 340, 120, 45, 280, 180, 30, 260, 160];
  const boxes = colors.map((hue, i) => `
    <div style="aspect-ratio:1;background:hsl(${hue},60%,70%);border-radius:var(--radius);box-shadow:var(--shadow);display:flex;align-items:center;justify-content:center;color:#fff;font-size:1.2rem;font-weight:600;">
      Image ${i + 1}
    </div>`).join("\n");

  return `<section style="padding:48px 32px;">
  <div style="max-width:1200px;margin:0 auto;">
    <h2 style="text-align:center;margin-bottom:32px;color:var(--text);">Gallery</h2>
    <div class="grid-3" style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
${boxes}
    </div>
  </div>
</section>`;
}
