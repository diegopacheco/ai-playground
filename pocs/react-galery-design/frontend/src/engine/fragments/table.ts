export function generateTable(): string {
  const rows = [
    { id: 1, name: "Project Alpha", status: "Active", date: "2026-01-15", amount: "$12,400" },
    { id: 2, name: "Project Beta", status: "Pending", date: "2026-01-22", amount: "$8,750" },
    { id: 3, name: "Project Gamma", status: "Active", date: "2026-02-01", amount: "$23,100" },
    { id: 4, name: "Project Delta", status: "Completed", date: "2026-02-05", amount: "$5,200" },
    { id: 5, name: "Project Epsilon", status: "Active", date: "2026-02-07", amount: "$17,800" },
  ];

  const trs = rows.map((r, i) => `
      <tr style="background:${i % 2 === 0 ? "transparent" : "rgba(0,0,0,0.03)"};">
        <td style="padding:12px 16px;border-bottom:1px solid rgba(0,0,0,0.08);">${r.id}</td>
        <td style="padding:12px 16px;border-bottom:1px solid rgba(0,0,0,0.08);">${r.name}</td>
        <td style="padding:12px 16px;border-bottom:1px solid rgba(0,0,0,0.08);"><span style="padding:4px 12px;border-radius:999px;font-size:0.8rem;background:${r.status === "Active" ? "var(--primary)" : r.status === "Completed" ? "var(--secondary)" : "#f59e0b"};color:#fff;">${r.status}</span></td>
        <td style="padding:12px 16px;border-bottom:1px solid rgba(0,0,0,0.08);">${r.date}</td>
        <td style="padding:12px 16px;border-bottom:1px solid rgba(0,0,0,0.08);">${r.amount}</td>
      </tr>`).join("\n");

  return `<section style="padding:48px 32px;">
  <div style="max-width:1200px;margin:0 auto;">
    <h2 style="margin-bottom:24px;color:var(--text);">Data Overview</h2>
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;background:var(--bg);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow);">
        <thead>
          <tr style="background:var(--primary);color:#fff;">
            <th style="padding:12px 16px;text-align:left;">ID</th>
            <th style="padding:12px 16px;text-align:left;">Name</th>
            <th style="padding:12px 16px;text-align:left;">Status</th>
            <th style="padding:12px 16px;text-align:left;">Date</th>
            <th style="padding:12px 16px;text-align:left;">Amount</th>
          </tr>
        </thead>
        <tbody>
${trs}
        </tbody>
      </table>
    </div>
  </div>
</section>`;
}
