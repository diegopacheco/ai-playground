export function generateFeatures(): string {
  const items = [
    { icon: "\u26A1", title: "Lightning Fast", desc: "Optimized for speed and performance from the ground up." },
    { icon: "\u{1F512}", title: "Secure", desc: "Enterprise-grade security to keep your data safe." },
    { icon: "\u{1F4CA}", title: "Analytics", desc: "Deep insights and real-time analytics at your fingertips." },
    { icon: "\u{1F310}", title: "Global Scale", desc: "Deploy worldwide with built-in CDN and edge computing." },
  ];

  const grid = items.map((f) => `
    <div style="padding:32px;background:var(--bg);border-radius:var(--radius);box-shadow:var(--shadow);text-align:center;">
      <div style="font-size:2.5rem;margin-bottom:16px;">${f.icon}</div>
      <h3 style="margin-bottom:8px;color:var(--text);">${f.title}</h3>
      <p style="color:var(--secondary);font-size:0.9rem;">${f.desc}</p>
    </div>`).join("\n");

  return `<section style="padding:48px 32px;">
  <div style="max-width:1200px;margin:0 auto;">
    <h2 style="text-align:center;margin-bottom:32px;color:var(--text);">Features</h2>
    <div class="grid-2" style="display:grid;grid-template-columns:repeat(2,1fr);gap:24px;">
${grid}
    </div>
  </div>
</section>`;
}
