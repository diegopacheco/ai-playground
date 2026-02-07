export function generatePricing(): string {
  const plans = [
    { name: "Basic", price: "$9", features: ["5 Projects", "1GB Storage", "Email Support", "Basic Analytics"] },
    { name: "Pro", price: "$29", features: ["25 Projects", "10GB Storage", "Priority Support", "Advanced Analytics", "Custom Domain"] },
    { name: "Enterprise", price: "$99", features: ["Unlimited Projects", "100GB Storage", "24/7 Support", "Full Analytics", "Custom Domain", "SSO & SAML"] },
  ];

  const cards = plans.map((p, i) => {
    const highlight = i === 1;
    return `
    <div class="card" style="padding:32px;background:var(--bg);border-radius:var(--radius);box-shadow:var(--shadow);text-align:center;${highlight ? "border:2px solid var(--primary);transform:scale(1.05);" : "border:1px solid rgba(0,0,0,0.1);"}">
      <h3 style="margin-bottom:8px;color:var(--text);">${p.name}</h3>
      <div style="font-size:2.5rem;font-weight:800;color:var(--primary);margin:16px 0;">${p.price}<span style="font-size:1rem;font-weight:400;color:var(--secondary);">/mo</span></div>
      <ul style="list-style:none;padding:0;margin:24px 0;text-align:left;">
        ${p.features.map((f) => `<li style="padding:8px 0;border-bottom:1px solid rgba(0,0,0,0.06);color:var(--text);font-size:0.9rem;">${f}</li>`).join("\n        ")}
      </ul>
      <button style="width:100%;padding:12px;font-size:1rem;">Choose Plan</button>
    </div>`;
  }).join("\n");

  return `<section style="padding:48px 32px;">
  <div style="max-width:1000px;margin:0 auto;">
    <h2 style="text-align:center;margin-bottom:32px;color:var(--text);">Pricing</h2>
    <div class="pricing-grid grid-3" style="display:grid;grid-template-columns:repeat(3,1fr);gap:24px;align-items:center;">
${cards}
    </div>
  </div>
</section>`;
}
